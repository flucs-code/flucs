#pragma once

#include <cupy/complex.cuh>

// Deal with float types
#ifdef DOUBLE_PRECISION
    #define FLUCS_FLOAT double
    #define flucs_fabs(x) fabs(x)
    #define FLUCS_COMPLEX_FLOAT_EQUIV double2
#else
    #define FLUCS_FLOAT float
    #define flucs_fabs(x) fabsf(x)
    #define FLUCS_COMPLEX_FLOAT_EQUIV float2
#endif

#define FLUCS_COMPLEX complex<FLUCS_FLOAT>

#define FLOAT_ONE ((FLUCS_FLOAT)1.0)
#define COMPLEX_ONE FLUCS_COMPLEX(FLOAT_ONE, 0)

#include "flucs/solvers/fourier/fourier_system_utilities.cuh"
#include "flucs/solvers/fourier/fourier_system_indexing.cuh"
#include "flucs/solvers/fourier/fourier_system_reductions.cuh"

extern "C" {


// Precomputed matrices stored in constant memory
__constant__ FLUCS_COMPLEX* rhs_precomp = NULL;
__constant__ FLUCS_COMPLEX* lhs_precomp = NULL;
__constant__ FLUCS_COMPLEX* inverse_lhs_precomp = NULL;


// Gets the linear matrix for a single mode.
// This must be implemented by the user.
__device__ void get_linear_matrix(const int index,
                                  const FLUCS_FLOAT dt,
                                  FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]);

// Finds the nonlinear terms for the current time
// step and adds them to rhs_fields.
// Must be implemented by the user.
__device__ void add_nonlinear_terms(const int index,
                                    const FLUCS_FLOAT dt,
                                    const int current_step,
                                    const FLUCS_FLOAT AB0,
                                    const FLUCS_FLOAT AB1,
                                    const FLUCS_FLOAT AB2,
                                    const FLUCS_COMPLEX* dft_bits,
                                    FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS]);


// Returns the full (for all modes) linear matrix.
// Matrix is assumed to be contiguous with shape (NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, index)
__global__ void compute_linear_matrix(const FLUCS_FLOAT dt, FLUCS_COMPLEX* linear_matrix){
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_linear_matrix(index, dt, matrix);

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            linear_matrix[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = matrix[i][j];
        }
    }
}


// Precomputes the rhs and inverse_lhs matrices.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX inverse_lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    { // Compiler hint
        const FLUCS_FLOAT ALPHA_DT = ALPHA*dt;
        const FLUCS_FLOAT ALPHAMINUS1_DT = (ALPHA - 1)*dt;
    
        FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
        get_linear_matrix(index, dt, matrix);

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){

                rhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] =\
                    (FLUCS_FLOAT)(i == j) + ALPHAMINUS1_DT * matrix[i][j];

                lhs[i][j] = (FLUCS_FLOAT)(i == j) + ALPHA_DT * matrix[i][j];
            }
        }
    }

    invert_matrix(lhs, inverse_lhs);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            inverse_lhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = inverse_lhs[i][j];
        }
    }
}


// Called right at the end of a time step,
// combines the linear matrices and nonlinear
// terms to find the fields at the current time step.
__global__ void finish_step(const FLUCS_FLOAT dt,
                            const int current_step,
                            const FLUCS_FLOAT AB0,
                            const FLUCS_FLOAT AB1,
                            const FLUCS_FLOAT AB2,
                            const FLUCS_COMPLEX* previous_fields,
                            const FLUCS_COMPLEX* dft_bits,
                            FLUCS_COMPLEX* current_fields){

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS];

#ifdef PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            sum += rhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * previous_fields[index + j*HALFUNPADDEDSIZE];
        }
        rhs_fields[i] = sum;
    }

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_step, AB0, AB1, AB2, dft_bits, rhs_fields);
#endif


    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            sum += inverse_lhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * rhs_fields[j];
        }
        current_fields[index + i*HALFUNPADDEDSIZE] = sum;
    }
#else // not PRECOMPUTE_LINEAR_MATRIX

    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    // Help the compiler a bit with the registers
    {
        FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
        get_linear_matrix(index, dt, matrix);

        const FLUCS_FLOAT ALPHA_DT = ALPHA*dt;
        const FLUCS_FLOAT ALPHAMINUS1_DT = (ALPHA - 1)*dt;

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                lhs[i][j] = (FLUCS_FLOAT)(i == j) + ALPHA_DT * matrix[i][j];

                sum += ( (FLUCS_FLOAT)(i == j) + ALPHAMINUS1_DT * matrix[i][j] )\
                    * previous_fields[index + j*HALFUNPADDEDSIZE];
            }

            rhs_fields[i] = sum;
        }
    }

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_step, AB0, AB1, AB2, dft_bits, rhs_fields);
#endif

    FLUCS_COMPLEX result[NUMBER_OF_FIELDS];
    gaussian_elimination(lhs, result, rhs_fields);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        current_fields[index + i*HALFUNPADDEDSIZE] = result[i];
    }

#endif // PRECOMPUTE_LINEAR_MATRIX
}

} // extern "C"
