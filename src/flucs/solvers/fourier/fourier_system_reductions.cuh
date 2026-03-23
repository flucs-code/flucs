// Various useful sums/averages over various dimensions
#pragma once


// C++ section, contains various templated and/or overloaded functions

// Atomic addition for complex numbers
// (CURRENTLY NOT USED)
// __device__ __forceinline__
// atomicAdd_complex(FLUCS_COMPLEX* address, FLUCS_COMPLEX val) {
//     // Cast complex to float to allow easy access to real and imag parts
//     FLUCS_FLOAT* ptr = reinterpret_cast<FLUCS_FLOAT*>(address);
//
//     atomicAdd(&ptr[0], val.real());
//     atomicAdd(&ptr[1], val.imag());
// }


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
template <size_t N, typename T, typename... Functors>
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
template <size_t N, typename T, typename... Functors>
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
template <typename T, T multiplier>
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
template <size_t N, bool is_half_axis, typename T, typename... Functors>
__device__ __forceinline__
void multiply_and_sum_last_axis(
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    const size_t ix  = blockIdx.x;
    const size_t tid = threadIdx.x;

    T sum = 0;

    // Grid-stride loop over contiguous axis
    for (size_t iy = tid; iy < N; iy += blockDim.x) {
        sum += ((is_half_axis && iy > 0) ? (FLUCS_FLOAT)2.0 : (FLUCS_FLOAT)1.0) * multiply_at<N, T>(ix * N + iy, array_functors...);
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
template <size_t N, bool is_half_axis, typename T, typename... Functors>
__device__ __forceinline__
void add_and_sum_last_axis(
    const T multiplier,
    T* __restrict__ output,
    Functors... array_functors)
{
    const size_t ix  = blockIdx.x;
    const size_t tid = threadIdx.x;

    T sum = 0;

    // Grid-stride loop over contiguous axis
    for (size_t iy = tid; iy < N; iy += blockDim.x) {
        sum += ((is_half_axis && iy > 0) ? (FLUCS_FLOAT)2.0 : (FLUCS_FLOAT)1.0) * add_at<N, T>(ix * N + iy, array_functors...);
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
// End of C++ section


extern "C" {

// FOURIER SPACE

// Sum over the last axis of an (M, HALF_NY) array.
__global__
void last_axis_sum_half_ny(
    const FLUCS_COMPLEX* __restrict__ input,
    FLUCS_COMPLEX* __restrict__ output) {

    multiply_and_sum_last_axis<HALF_NY, true>(COMPLEX_ONE,
                                                       output,
                                                       NOP_Functor<FLUCS_COMPLEX>{input});
}

// Sum over the last axis of an (M, NX) array.
__global__
void last_axis_sum_nx(
    const FLUCS_COMPLEX* __restrict__ input,
    FLUCS_COMPLEX* __restrict__ output) {

    multiply_and_sum_last_axis<NX, false>(COMPLEX_ONE,
                                                  output,
                                                  NOP_Functor<FLUCS_COMPLEX>{input});
}

// Sum over the last axis of an (M, NZ) array.
__global__
void last_axis_sum_nz(
    const FLUCS_COMPLEX* __restrict__ input,
    FLUCS_COMPLEX* __restrict__ output) {

    multiply_and_sum_last_axis<NZ, false>(COMPLEX_ONE,
                                                  output,
                                                  NOP_Functor<FLUCS_COMPLEX>{input});
}


// REAL UNPADDED
__global__
void real_last_axis_sum_half_ny(
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis<HALF_NY, true>(FLOAT_ONE,
                                                       output,
                                                       NOP_Functor<FLUCS_FLOAT>{input});
}

// Sum over the last axis of an (M, NX) array.
__global__
void real_last_axis_sum_nx(
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis<NX, false>(FLOAT_ONE,
                                                  output,
                                                  NOP_Functor<FLUCS_FLOAT>{input});
}

// Sum over the last axis of an (M, NZ) array.
__global__
void real_last_axis_sum_nz(
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis<NZ, false>(FLOAT_ONE,
                                                  output,
                                                  NOP_Functor<FLUCS_FLOAT>{input});
}
// REAL PADDED SPACE

// Average over the last axis of an (M, PADDED_NX) array.
__global__
void last_axis_average_padded_nx(
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis<PADDED_NX, false>(FLOAT_ONE / (FLUCS_FLOAT)PADDED_NX,
                                                       output,
                                                       NOP_Functor<FLUCS_FLOAT>{input});
}

// Average over the last axis of an (M, PADDED_NY) array.
__global__
void last_axis_average_padded_ny(
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis<PADDED_NY, false>(FLOAT_ONE / (FLUCS_FLOAT)PADDED_NY,
                                                       output,
                                                       NOP_Functor<FLUCS_FLOAT>{input});
}

// Average over the last axis of an (M, PADDED_NZ) array.
__global__
void last_axis_average_padded_nz(
    const FLUCS_FLOAT* __restrict__ input,
    FLUCS_FLOAT* __restrict__ output) {

    multiply_and_sum_last_axis<PADDED_NZ, false>(FLOAT_ONE / (FLUCS_FLOAT)PADDED_NZ,
                                                       output,
                                                       NOP_Functor<FLUCS_FLOAT>{input});
}

} // extern "C"
