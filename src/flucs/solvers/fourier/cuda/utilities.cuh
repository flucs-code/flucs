#pragma once


// ---------------------------------- //
// Useful utilities missing from CUDA //
// ---------------------------------- //
 
// No-op conj for floats
__device__ __forceinline__
FLUCS_FLOAT conj(FLUCS_FLOAT x)  { return x; }

// Atomic addition for complex numbers
__device__ __forceinline__
void atomicAdd(FLUCS_COMPLEX* address, FLUCS_COMPLEX val) {
    // Cast complex to float to allow easy access to real and imag parts
    FLUCS_FLOAT* ptr = reinterpret_cast<FLUCS_FLOAT*>(address);

    atomicAdd(&ptr[0], val.real());
    atomicAdd(&ptr[1], val.imag());
}

// Fixing the annoyances with templated shared memory
template <typename T>
__device__ __forceinline__ T* templated_shared_memory() {
    extern __shared__ unsigned char shared[];
    return reinterpret_cast<T*>(shared);
}

__device__ float atomicMaxFloat(float* addr, float value) {
    int* address_as_int = (int*) addr;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ double atomicMaxFloat(double* addr, double value)
{
    unsigned long long* address_as_ull =
        reinterpret_cast<unsigned long long*>(addr);

    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;

        double assumed_val = __longlong_as_double(assumed);
        double max_val = fmax(value, assumed_val);

        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(max_val)
        );

    } while (assumed != old);

    return __longlong_as_double(old);
}


// --------------------- //
// Simple linear algebra //
// --------------------- //

// Solves Ax = b using pivotless Gaussian elimination.
// Both A and b are lost in the process!
// The output is x = A^-1 b.
// It can be used in-place, i.e., x and b can be the same.
template<int N, typename T>
__device__ __forceinline__
void gaussian_elimination_inplace(T A[N][N],
                                  T x[N],
                                  T b[N]) {

    // Bog-standard and basic
    // forward elimination then back substitution
    
    #pragma unroll
    for (int k = 0; k < N; k++) {
        T inv_pivot = COMPLEX_ONE / A[k][k];

        #pragma unroll
        for (int i = k + 1; i < N; i++) {
            T factor = A[i][k] * inv_pivot;

            A[i][k] = 0;
            #pragma unroll
            for (int j = k + 1; j < N; j++)
                A[i][j] -= factor * A[k][j];

            b[i] -= factor * b[k];
        }
    }

    #pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        T sum = b[i];

        #pragma unroll
        for (int j = i + 1; j < N; j++)
            sum -= A[i][j] * x[j];

        x[i] = sum / A[i][i];
    }
}

// Wrapper around gaussian_elimination_inplace
// that does not destroy A and b.
template<int N, typename T>
__device__ __forceinline__
void gaussian_elimination(T A[N][N],
                          T x[N],
                          T b[N]) {

    T A_copy[N][N], b_copy[N];

    #pragma unroll
    for (int i = 0; i < N; i++) {
        
        b_copy[i] = b[i];
        #pragma unroll
        for (int j = 0; j < N; j++) {
            A_copy[i][j] = A[i][j];
        }
    }

    gaussian_elimination_inplace(A_copy, x, b_copy);
}

// Solves AX = B using pivotless Gaussian elimination.
// Both A and B are lost in the process!
// The output is X = A^-1 B.
// It can be used in-place, i.e., X and B can be the same.
template<int N, int M, typename T>
__device__ __forceinline__
void gaussian_elimination_inplace(T A[N][N],
                                  T X[N][M],
                                  T B[N][M]) {

    // Bog-standard and basic
    // forward elimination then back substitution
    
    #pragma unroll
    for (int k = 0; k < N; k++) {
        T inv_pivot = COMPLEX_ONE / A[k][k];

        #pragma unroll
        for (int i = k + 1; i < N; i++) {
            T factor = A[i][k] * inv_pivot;

            A[i][k] = 0;
            #pragma unroll
            for (int j = k + 1; j < N; j++)
                A[i][j] -= factor * A[k][j];

            // Apply same row operation to every RHS column.
            // Don't unroll in case we spill
            #pragma unroll 1
            for (int c = 0; c < M; c++) {
                B[i][c] -= factor * B[k][c];
            }
        }
    }

    #pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        T inv_diag = COMPLEX_ONE / A[i][i];

        // Don't unroll in case we spill
        #pragma unroll 1
        for (int c = 0; c < M; c++) {

            T sum = B[i][c];

            #pragma unroll
            for (int j = i + 1; j < N; j++)
                sum -= A[i][j] * X[j][c];

            X[i][c] = sum * inv_diag;
        }
    }
}

// Wrapper around gaussian_elimination_inplace
// that does not destroy A and B.
template<int N, int M, typename T>
__device__ __forceinline__
void gaussian_elimination(T A[N][N],
                          T X[N][M],
                          T B[N][M]) {

    T A_copy[N][N], B_copy[N][M];

    #pragma unroll
    for (int i = 0; i < N; i++) {
        
        #pragma unroll
        for (int j = 0; j < N; j++) {
            A_copy[i][j] = A[i][j];
        }

        #pragma unroll
        for (int j = 0; j < M; j++) {
            B_copy[i][j] = B[i][j];
        }
    }

    gaussian_elimination_inplace<N, M>(A_copy, X, B_copy);
}

// Finds the inverse of a matrix by Gaussian elimination of
// AX = Identity
// A is destroyed in the process
template<int N, typename T>
__device__ __forceinline__
void invert_matrix_inplace(T A[N][N], T inv_A[N][N]) {
    T identity[N][N];

    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N; j++) {
            identity[i][j] = (i == j) ? T(1) : T(0);
        }
    }

    gaussian_elimination_inplace<N, N>(A, inv_A, identity);
}

// Wrapper around invert_matrix_inplace
// that does not destroy A and B.
template<int N, typename T>
__device__ __forceinline__
void invert_matrix(const T A[N][N], T inv_A[N][N]) {
    T A_copy[N][N];

    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N; j++) {
            A_copy[i][j] = A[i][j];
        }
    }
    invert_matrix_inplace(A_copy, inv_A);
}

// Naive unrolled matrix multiplication for C = AB
template<int N, int M, int K, typename T>
__device__ __forceinline__
void small_matmul(const T A[N][K], const T B[K][M], T C[N][M]) {

    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < M; j++) {
            T sum = 0;

            #pragma unroll
            for (int k = 0; k < K; k++) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }
}

