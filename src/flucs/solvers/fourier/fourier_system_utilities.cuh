#pragma once

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

// Solves Ax = b using pivotless Gaussian elimination.
// Both A and b are lost in the process!
// The output is x = A^-1 b.
template<int N, typename T>
__device__ __forceinline__
void gaussian_elimination(T A[N][N],
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

            #pragma unroll
            for (int j = k; j < N; j++)
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

template<int N, typename T>
__device__ __forceinline__
void invert_matrix(const T A[N][N], T inv_A[N][N]) {
    T x[N];
    T b[N];

    // Essentially, do Gaussian elimination on each column of
    // the identity matrix.
    // Don't unroll in case we push those registers too hard.
    for (int col = 0; col < N; col++) {
        // Temporary copy of A
        T tmp_A[N][N];
        #pragma unroll
        for (int i = 0; i < N; i++)
            #pragma unroll
            for (int j = 0; j < N; j++)
                tmp_A[i][j] = A[i][j];

        // Initialize b as the col-th column of the identity
        #pragma unroll
        for (int i = 0; i < N; i++)
            b[i] = (i == col) ? T(1) : T(0);

        gaussian_elimination(tmp_A, x, b);

        // Store the result as the col-th column of inv_A
        #pragma unroll
        for (int i = 0; i < N; i++)
            inv_A[i][col] = x[i];
    }
}

