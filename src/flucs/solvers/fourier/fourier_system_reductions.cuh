// Various useful sums/averages over various dimensions
#pragma once


// C++ section, contains various templated and/or overloaded functions

// Sums over warps
__device__ __forceinline__
FLUCS_FLOAT warp_sum(FLUCS_FLOAT v)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }

    return v;
}

__device__ __forceinline__
FLUCS_COMPLEX warp_sum(FLUCS_COMPLEX v)
{
    FLUCS_COMPLEX_FLOAT_EQUIV x = *reinterpret_cast<FLUCS_COMPLEX_FLOAT_EQUIV*>(&v);

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        x.x += __shfl_down_sync(0xffffffff, x.x, offset);
        x.y += __shfl_down_sync(0xffffffff, x.y, offset);
    }

    return *reinterpret_cast<FLUCS_COMPLEX*>(&x);
}

// Helper function for expanding varargs into a sum
template <typename T, typename... Functors>
__device__ __forceinline__
T add_at(size_t index, Functors... array_functors)
{
    T values[] = { array_functors(index)... };

    T result = values[0];
    #pragma unroll
    for (int i = 1; i < sizeof...(Functors); ++i)
        result += values[i];

    return result;
}

// Helper function for expanding varargs into a product
template <typename T, typename... Functors>
__device__ __forceinline__
T multiply_at(size_t index, Functors... array_functors)
{
    T values[] = { array_functors(index)... };

    T result = values[0];
    #pragma unroll
    for (int i = 1; i < sizeof...(Functors); ++i)
        result *= values[i];

    return result;
}

// Array functors

// NOP functor, it does nothing
template <typename T>
struct NOP_Functor {
    const T* __restrict__ array;
    __device__ __forceinline__ T operator()(size_t index) const {
        return array[index];
    }
};

// Conjugate functor
struct CC_Functor {
    const FLUCS_COMPLEX* __restrict__ array;
    __device__ __forceinline__ FLUCS_COMPLEX operator()(size_t index) const {
        FLUCS_COMPLEX val = array[index];
        return FLUCS_COMPLEX(val.real(), -val.imag());
    }
};

// Abs2 functor with a multiplier
struct Abs2_Functor {
    const FLUCS_COMPLEX* __restrict__ array;
    const FLUCS_FLOAT multiplier;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        FLUCS_COMPLEX val = array[index];
        return multiplier * (val.real()*val.real() + val.imag()*val.imag());
    }
};

// Functor for multiplying by a constant
template <typename T>
struct ConstMultiplier_Functor {
    const T* __restrict__ array;
    __device__ __forceinline__ T operator()(size_t index) const {
        return multiplier * array[index];
    }
};

// d/dx functor for standard 3D Fourier space
struct Dx_Functor {
    const FLUCS_COMPLEX* __restrict__ array;
    __device__ __forceinline__ FLUCS_COMPLEX operator()(size_t index) const {
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikx = indices.ikx;

        return FLUCS_COMPLEX(0, kx_from_ikx(ikx)) * array[index];
    }
};

// d/dy functor for standard 3D Fourier space
struct Dy_Functor {
    const FLUCS_COMPLEX* __restrict__ array;
    __device__ __forceinline__ FLUCS_COMPLEX operator()(size_t index) const {
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t iky = indices.iky;

        return FLUCS_COMPLEX(0, ky_from_iky(iky)) * array[index];
    }
};

// d/dz functor for standard 3D Fourier space
struct Dz_Functor {
    const FLUCS_COMPLEX* __restrict__ array;
    __device__ __forceinline__ FLUCS_COMPLEX operator()(size_t index) const {
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikz = indices.ikz;

        return FLUCS_COMPLEX(0, kz_from_ikz(ikz)) * array[index];
    }
};

// del_perp^2 functor for standard 3D Fourier space
struct DelPerp2_Functor {
    const FLUCS_COMPLEX* __restrict__ array;
    __device__ __forceinline__ FLUCS_COMPLEX operator()(size_t index) const {
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        return -(kx*kx + ky*ky) * array[index];
    }
};

/*
 * A kernel that takes an arbitrary number of array functors,
 * multiplies them together, then reinterprets the result
 * as an (M, N)-shaped array, reduces over the contiguous axis,
 * and outputs the resulting array (of length M) multiplied
 * by multiplier.
 *
 * If is_half_axis is true, all elements with nonzero index along
 * the summed axis (the one of length N) are multiplied by an
 * additional factor of 2, which is useful for reducing Fourier-space
 * quantities. NB: this factor of 2 produces crap unless one
 * sums over the M axis, too, or if the M axis is a real one.
 *
 * The kernel must be invoked with a grid size equal to M and uses
 * 32 * sizeof(T) shared memory.
 *
 * There are no restrictions on the block size, so feel free to optimise.
 * Block sizes larger than N are typically detrimental to performance.
 *
 * This is not meant to be invoked directly but rather through defining
 * specific kernels that deal with a specific set of inputs.
 *
 * Some examples are given below.
 */
template <typename T, typename... Functors>
__device__ __forceinline__
void multiply_and_sum_last_axis(
    const size_t N,
    const bool is_half_axis,
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    const size_t ix  = blockIdx.x;
    const size_t tid = threadIdx.x;

    T sum = 0;

    // Grid-stride loop over contiguous axis
    for (size_t iy = tid; iy < N; iy += blockDim.x) {
        sum += ((is_half_axis && iy > 0) ? (FLUCS_FLOAT)2.0 : (FLUCS_FLOAT)1.0) * multiply_at<T>(ix * N + iy, array_functors...);
    }

    // Warp-level reduction
    sum = warp_sum(sum);

    // CUDA allows at most 32 warps per block
    __shared__ T warp_sums[32];

    // Move all partial sums to the first warp
    if ((tid & 31) == 0)
        warp_sums[tid >> 5] = sum;

    __syncthreads();

    // Final reduction done by the first warp
    if (tid < warpSize) {
        T v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0;

        v = warp_sum(v);

        if (tid == 0)
            output[ix] = v * multiplier;
    }
}

// Same as the product kernel but now we add the functors
// element-wise instead of multiplying them.
template <typename T, typename... Functors>
__device__ __forceinline__
void add_and_sum_last_axis(
    const size_t N,
    const bool is_half_axis,
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    const size_t ix  = blockIdx.x;
    const size_t tid = threadIdx.x;

    T sum = 0;

    // Grid-stride loop over contiguous axis
    for (size_t iy = tid; iy < N; iy += blockDim.x) {
        sum += ((is_half_axis && iy > 0) ? (FLUCS_FLOAT)2.0 : (FLUCS_FLOAT)1.0) * add_at<T>(ix * N + iy, array_functors...);
    }

    // Warp-level reduction
    sum = warp_sum(sum);

    // CUDA allows at most 32 warps per block
    __shared__ T warp_sums[32];

    // Move all partial sums to the first warp
    if ((tid & 31) == 0)
        warp_sums[tid >> 5] = sum;

    __syncthreads();

    // Final reduction done by the first warp
    if (tid < warpSize) {
        T v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0;

        v = warp_sum(v);

        if (tid == 0)
            output[ix] = v * multiplier;
    }
}

template <typename T, typename... Functors>
__device__ __forceinline__
void add_and_shell_sum(
    const size_t nkperp,
    const FLUCS_FLOAT kperp_min,
    const FLUCS_FLOAT kperp_max,
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    // kperp bins are stored in shared memory
    T* kperp_bins = templated_shared_memory<T>();

    const FLUCS_FLOAT inv_dkperp = ((FLUCS_FLOAT)nkperp) / (kperp_max - kperp_min);

    // One block per kz
    const size_t ikz  = blockIdx.x;

    const size_t tid = threadIdx.x;

    // First, zero out shared mem
    for (size_t bin_index = tid; bin_index < nkperp; bin_index += blockDim.x) {
        kperp_bins[bin_index] = 0;
    }

    __syncthreads();


    // Each thread reads data from global memory in a contiguous way
    for (size_t perp_index = tid; perp_index < NX*HALF_NY; perp_index += blockDim.x) {

        // Convert perp index to ikx and iky
        indices3d_t indices = get_indices3d<1, NX, HALF_NY>(perp_index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp = flucs_sqrt(kx*kx + ky*ky);
        const FLUCS_FLOAT bin_index_float = (kperp - kperp_min) * inv_dkperp;

        // Evaluates true if kperp < KPERP_MIN
        // Need to compare the float index as casting negative floats to ints
        // rounds up rather than down
        if (bin_index_float < 0)
            continue;

        // Now can safely cast to int, which rounds down
        // and we can check if kperp > KPERP_MAX
        const int bin_index = (int)(bin_index_float);
        if (bin_index >= nkperp)
            continue;


        // Construct full indices
        const size_t mode_index = index_from_3d<NZ, NX, HALF_NY>(ikz, ikx, iky);
        const T mode = add_at<T>(mode_index, array_functors...);

        // Add mode to shared-memory bin
        atomicAdd(&kperp_bins[bin_index], mode);

        // If ky > 0, need to add the missing conjugate mode
        if (iky > 0) {
            const size_t conj_mode_index = index_from_3d<NZ, NX, HALF_NY>(
                (ikz == 0 ? 0 : NZ - ikz),
                (ikx == 0 ? 0 : NX - ikx),
                iky
            );
            const T conj_mode = add_at<T>(conj_mode_index, array_functors...);
            atomicAdd(&kperp_bins[bin_index], conj(conj_mode));
        }

        
    }

    __syncthreads();


    // Finally, write the output from shared into global memory
    for (size_t bin_index = tid; bin_index < nkperp; bin_index += blockDim.x) {
        output[bin_index + ikz*nkperp] = kperp_bins[bin_index] * multiplier;
    }
 
}

template <typename T, typename... Functors>
__device__ __forceinline__
void multiply_and_shell_sum(
    const size_t nkperp,
    const FLUCS_FLOAT kperp_min,
    const FLUCS_FLOAT kperp_max,
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    // kperp bins are stored in shared memory
    T* kperp_bins = templated_shared_memory<T>();

    const FLUCS_FLOAT inv_dkperp = ((FLUCS_FLOAT)nkperp) / (kperp_max - kperp_min);

    // One block per kz
    const size_t ikz  = blockIdx.x;

    const size_t tid = threadIdx.x;

    // First, zero out shared mem
    for (size_t bin_index = tid; bin_index < nkperp; bin_index += blockDim.x) {
        kperp_bins[bin_index] = 0;
    }

    __syncthreads();


    // Each thread reads data from global memory in a contiguous way
    for (size_t perp_index = tid; perp_index < NX*HALF_NY; perp_index += blockDim.x) {

        // Convert perp index to ikx and iky
        indices3d_t indices = get_indices3d<1, NX, HALF_NY>(perp_index);
        const size_t ikx = indices.ikx;
        const size_t iky = indices.iky;

        const FLUCS_FLOAT kx = kx_from_ikx(ikx);
        const FLUCS_FLOAT ky = ky_from_iky(iky);
        const FLUCS_FLOAT kperp = flucs_sqrt(kx*kx + ky*ky);
        const FLUCS_FLOAT bin_index_float = (kperp - kperp_min) * inv_dkperp;

        // Evaluates true if kperp < KPERP_MIN
        // Need to compare the float index as casting negative floats to ints
        // rounds up rather than down
        if (bin_index_float < 0)
            continue;

        // Now can safely cast to int, which rounds down
        // and we can check if kperp > KPERP_MAX
        const int bin_index = (int)(bin_index_float);
        if (bin_index >= nkperp)
            continue;


        // Construct full indices
        const size_t mode_index = index_from_3d<NZ, NX, HALF_NY>(ikz, ikx, iky);
        const T mode = multiply_at<T>(mode_index, array_functors...);

        // Add mode to shared-memory bin
        atomicAdd(&kperp_bins[bin_index], mode);

        // If ky > 0, need to add the missing conjugate mode
        if (iky > 0) {
            const size_t conj_mode_index = index_from_3d<NZ, NX, HALF_NY>(
                (ikz == 0 ? 0 : NZ - ikz),
                (ikx == 0 ? 0 : NX - ikx),
                iky
            );
            const T conj_mode = multiply_at<T>(conj_mode_index, array_functors...);
            atomicAdd(&kperp_bins[bin_index], conj(conj_mode));
        }

        
    }

    __syncthreads();


    // Finally, write the output from shared into global memory contiguously
    for (size_t bin_index = tid; bin_index < nkperp; bin_index += blockDim.x) {
        output[bin_index + ikz*nkperp] = kperp_bins[bin_index] * multiplier;
    }
 
}

// Somewhat naive but gets the job done
// Reduces an (m, n, k) array down to (m, k) one
// Must be invoked with a block (block_size, )
// and a grid (m, (k + block_size - 1) // k)
template <typename T, typename... Functors>
__device__ __forceinline__
void add_and_sum_middle_axis(
    const size_t n,
    const size_t k,
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    const size_t iy = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(iy < k))
        return;

    T sum = 0;

    for (size_t ix = 0; ix < n; ix++) {
        sum += add_at<T>((ix + n * blockIdx.y) * k + iy, array_functors...);
    }
    
    output[k * blockIdx.y + iy] = sum * multiplier;
}

template <typename T, typename... Functors>
__device__ __forceinline__
void multiply_and_sum_middle_axis(
    const size_t n,
    const size_t k,
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    const size_t iy = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(iy < k))
        return;

    T sum = 0;

    for (size_t ix = 0; ix < n; ix++) {
        sum += multiply_at<T>((ix + n * blockIdx.y) * k + iy, array_functors...);
    }
    
    output[k * blockIdx.y + iy] = sum * multiplier;
}


// Templated reduction kernels
template <size_t N, bool is_half_axis, typename T_output, typename Functor, typename... InputArgs>
__global__
void simple_last_axis_sum(T_output* __restrict__ output, InputArgs... input_args) {
    multiply_and_sum_last_axis(
        N,
        is_half_axis,
        (T_output)FLOAT_ONE,
        output,
        Functor{input_args...}
    );
}

template <size_t N, size_t K, typename T_output, typename Functor, typename... InputArgs>
__global__
void simple_middle_axis_sum(T_output* __restrict__ output, InputArgs... input_args) {
    multiply_and_sum_middle_axis(
        N,
        K,
        (T_output)FLOAT_ONE,
        output,
        Functor{input_args...}
    );
}

// End of C++ section

extern "C" {

__global__
void last_axis_sum_float(
    const size_t N,
    const bool is_half_axis,
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis(
        N,
        is_half_axis,
        FLOAT_ONE,
        output,
        NOP_Functor<FLUCS_FLOAT>{input}
    );
}

__global__
void last_axis_sum_complex(
    const size_t N,
    const bool is_half_axis,
    const FLUCS_COMPLEX* __restrict__ input,
    FLUCS_COMPLEX* __restrict__ output) {

    multiply_and_sum_last_axis(
        N,
        is_half_axis,
        COMPLEX_ONE,
        output,
        NOP_Functor<FLUCS_COMPLEX>{input}
    );
}

__global__
void last_axis_average_float(
    const size_t N,
    const bool is_half_axis,
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis(
        N,
        is_half_axis,
        FLOAT_ONE / (FLUCS_FLOAT)N,
        output,
        NOP_Functor<FLUCS_FLOAT>{input}
    );
}

__global__
void last_axis_average_complex(
    const size_t N,
    const bool is_half_axis,
    const FLUCS_COMPLEX* __restrict__ input,
    FLUCS_COMPLEX* __restrict__ output) {

    multiply_and_sum_last_axis(
        N,
        is_half_axis,
        COMPLEX_ONE / (FLUCS_FLOAT)N,
        output,
        NOP_Functor<FLUCS_COMPLEX>{input}
    );
}

__global__
void shell_sum_float(
    const size_t nkperp,
    const FLUCS_FLOAT kperp_min,
    const FLUCS_FLOAT kperp_max,
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    add_and_shell_sum(
        nkperp,
        kperp_min,
        kperp_max,
        FLOAT_ONE,
        output,
        NOP_Functor<FLUCS_FLOAT>{input}
    );
}

__global__
void shell_sum_complex(
    const size_t nkperp,
    const FLUCS_FLOAT kperp_min,
    const FLUCS_FLOAT kperp_max,
    const FLUCS_COMPLEX* __restrict__ input,
    FLUCS_COMPLEX* __restrict__ output) {

    add_and_shell_sum<FLUCS_COMPLEX>(
        nkperp,
        kperp_min,
        kperp_max,
        COMPLEX_ONE,
        output,
        NOP_Functor<FLUCS_COMPLEX>{input}
    );
}

__global__
void shell_sum_field_squared(
    const size_t nkperp,
    const FLUCS_FLOAT kperp_min,
    const FLUCS_FLOAT kperp_max,
    const int field_index,
    const FLUCS_COMPLEX* __restrict__ fields,
    FLUCS_FLOAT* __restrict__ output) {

    const FLUCS_COMPLEX* field = fields + field_index * HALFUNPADDEDSIZE;

    add_and_shell_sum(
        nkperp,
        kperp_min,
        kperp_max,
        FLOAT_ONE,
        output,
        Abs2_Functor{field, FLOAT_ONE}
    );
}

} // extern "C"
