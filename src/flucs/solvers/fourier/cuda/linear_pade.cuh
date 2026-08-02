#pragma once

__device__
constexpr FLUCS_FLOAT EXP_PADE_COEFF(int k) {
    return ((FLUCS_FLOAT)(LINEAR_PADE_DEGREE - k + 1)) / (k * (2*LINEAR_PADE_DEGREE - k + 1));
}


#if LINEAR_PADE_DEGREE == 1
// [1,1] Pade
// lhs -> 1 + c_1 * A
// rhs -> 1 - c_1 * A
// where c_1 = 1/2
__device__ __forceinline__
void pade1_exponential(
    const FLUCS_COMPLEX A[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {

    constexpr FLUCS_FLOAT c1 = (FLUCS_FLOAT)0.5;

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            const FLUCS_COMPLEX Iij = (FLUCS_FLOAT)(i == j);

            lhs[i][j] = Iij + c1 * A[i][j];
            rhs[i][j] = Iij - c1 * A[i][j];
        }
    }
}
#endif // LINEAR_PADE_DEGREE == 1


#if LINEAR_PADE_DEGREE == 2
// [2,2] Pade
// lhs -> 1 + c_1 * A + c_2 * A^2
// rhs -> 1 - c_1 * A + c_2 * A^2
// where c_1 = 1/2, c_2 = 1/12
__device__ __forceinline__
void pade2_exponential(
    const FLUCS_COMPLEX A[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {

    FLUCS_COMPLEX A2[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A, A, A2);

    constexpr FLUCS_FLOAT c1 = (FLUCS_FLOAT)(1.0 / 2.0);
    constexpr FLUCS_FLOAT c2 = (FLUCS_FLOAT)(1.0 / 12.0);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            const FLUCS_COMPLEX Iij = (FLUCS_FLOAT)(i == j);

            lhs[i][j] = Iij + c1 * A[i][j] + c2 * A2[i][j];
            rhs[i][j] = Iij - c1 * A[i][j] + c2 * A2[i][j];
        }
    }
}
#endif // LINEAR_PADE_DEGREE == 2

#if LINEAR_PADE_DEGREE == 3
// [3,3] Pade
// lhs -> 1 + c_1 * A + c_2 * A^2 + c_3 * A^3
// rhs -> 1 - c_1 * A + c_2 * A^2 - c_3 * A^3
// where c_1 = 1/2, c_2 = 1/10, c_3 = 1/120
__device__ __forceinline__
void pade3_exponential(
    const FLUCS_COMPLEX A[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {

    FLUCS_COMPLEX A2[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX A3[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A, A,  A2);
    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A, A2, A3);

    constexpr FLUCS_FLOAT c1 = (FLUCS_FLOAT)(1.0 / 2.0);
    constexpr FLUCS_FLOAT c2 = (FLUCS_FLOAT)(1.0 / 10.0);
    constexpr FLUCS_FLOAT c3 = (FLUCS_FLOAT)(1.0 / 120.0);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            const FLUCS_COMPLEX Iij = (FLUCS_FLOAT)(i == j);

            lhs[i][j] = Iij + c1 * A[i][j] + c2 * A2[i][j] + c3 * A3[i][j];
            rhs[i][j] = Iij - c1 * A[i][j] + c2 * A2[i][j] - c3 * A3[i][j];
        }
    }
}
#endif // LINEAR_PADE_DEGREE == 3

#if LINEAR_PADE_DEGREE == 4
// [4,4] Pade
// lhs -> 1 + c_1 * A + c_2 * A^2 + c_3 * A^3 + c_4 * A^4
// rhs -> 1 - c_1 * A + c_2 * A^2 - c_3 * A^3 + c_4 * A^4
// where c_1 = 1/2, c_2 = 3/28, c_3 = 1/84, c_4 = 1/1680
__device__ __forceinline__
void pade4_exponential(
    const FLUCS_COMPLEX A[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {

    FLUCS_COMPLEX A2[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX A3[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX A4[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A,  A,  A2);
    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A,  A2, A3);
    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A2, A2, A4);

    constexpr FLUCS_FLOAT c1 = (FLUCS_FLOAT)(1.0 / 2.0);
    constexpr FLUCS_FLOAT c2 = (FLUCS_FLOAT)(3.0 / 28.0);
    constexpr FLUCS_FLOAT c3 = (FLUCS_FLOAT)(1.0 / 84.0);
    constexpr FLUCS_FLOAT c4 = (FLUCS_FLOAT)(1.0 / 1680.0);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            const FLUCS_COMPLEX Iij = (FLUCS_FLOAT)(i == j);

            lhs[i][j] = Iij
                        + c1 * A[i][j]
                        + c2 * A2[i][j]
                        + c3 * A3[i][j]
                        + c4 * A4[i][j];

            rhs[i][j] = Iij
                        - c1 * A[i][j]
                        + c2 * A2[i][j]
                        - c3 * A3[i][j]
                        + c4 * A4[i][j];
        }
    }

}
#endif // LINEAR_PADE_DEGREE == 4

#if LINEAR_PADE_DEGREE == 5
// [5,5] Pade
// lhs -> 1 + c_1 * A + c_2 * A^2 + c_3 * A^3 + c_4 * A^4 + c_5 * A^5
// rhs -> 1 - c_1 * A + c_2 * A^2 - c_3 * A^3 + c_4 * A^4 - c_5 * A^5
// where c_1 = 1/2, c_2 = 1/9, c_3 = 1/72, c_4 = 1/1008, c_5 = 1/30240
__device__ __forceinline__
void pade5_exponential(
    const FLUCS_COMPLEX A[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {

    FLUCS_COMPLEX A2[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX A3[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX A4[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX A5[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A,  A,  A2);
    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A,  A2, A3);
    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A2, A2, A4);
    small_matmul<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(A,  A4, A5);

    constexpr FLUCS_FLOAT c1 = (FLUCS_FLOAT)(1.0 / 2.0);
    constexpr FLUCS_FLOAT c2 = (FLUCS_FLOAT)(1.0 / 9.0);
    constexpr FLUCS_FLOAT c3 = (FLUCS_FLOAT)(1.0 / 72.0);
    constexpr FLUCS_FLOAT c4 = (FLUCS_FLOAT)(1.0 / 1008.0);
    constexpr FLUCS_FLOAT c5 = (FLUCS_FLOAT)(1.0 / 30240.0);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            const FLUCS_COMPLEX Iij = (FLUCS_FLOAT)(i == j);

            lhs[i][j] = Iij
                        + c1 * A[i][j]
                        + c2 * A2[i][j]
                        + c3 * A3[i][j]
                        + c4 * A4[i][j]
                        + c5 * A5[i][j];

            rhs[i][j] = Iij
                        - c1 * A[i][j]
                        + c2 * A2[i][j]
                        - c3 * A3[i][j]
                        + c4 * A4[i][j]
                        - c5 * A5[i][j];
        }
    }

}
#endif // LINEAR_PADE_DEGREE == 5

// Calculates the numerator and denominator of the Pade approximation
// of exp(A) and stores them in lhs and rhs, respectively. 
// Typically used with A = dt * linear_matrix for the implicit linear
// propagator.
__device__ __forceinline__
void pade_exponential(
    const FLUCS_COMPLEX A[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {
#if LINEAR_PADE_DEGREE == 1
    pade1_exponential(A, lhs, rhs);
#elif LINEAR_PADE_DEGREE == 2
    pade2_exponential(A, lhs, rhs);
#elif LINEAR_PADE_DEGREE == 3
    pade3_exponential(A, lhs, rhs);
#elif LINEAR_PADE_DEGREE == 4
    pade4_exponential(A, lhs, rhs);
#elif LINEAR_PADE_DEGREE == 5
    pade5_exponential(A, lhs, rhs);
#else
    #error "Unsupported LINEAR_PADE_DEGREE. Supported values are 1, 2, 3, 4, 5."
#endif
}
