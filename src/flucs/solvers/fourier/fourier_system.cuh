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


// rhs and inverse_lhs when using precomputed matrices.
__constant__ FLUCS_COMPLEX* rhs_precomp = NULL;
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
                                    FLUCS_COMPLEX* rhs_fields);

// Finds iterations matrices for a single mode.
__device__ void get_iteration_matrices(const int index,
                                       const FLUCS_FLOAT dt,
                                       FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
                                       FLUCS_COMPLEX inverse_lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]){

    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    get_linear_matrix(index, dt, matrix);
#if NUMBER_OF_FIELDS == 1
    // TODO: one field
#elif NUMBER_OF_FIELDS == 2
    // Hard-coded 2x2 inversion
    const FLUCS_FLOAT ALPHA_DT = ALPHA*dt;
    const FLUCS_FLOAT ALPHAMINUS1_DT = (ALPHA - 1)*dt;
    const FLUCS_FLOAT ONE = (FLUCS_FLOAT)1.0;

    rhs[0][0] = ONE + ALPHAMINUS1_DT*matrix[0][0];
    rhs[0][1] = ALPHAMINUS1_DT*matrix[0][1];
    rhs[1][0] = ALPHAMINUS1_DT*matrix[1][0];
    rhs[1][1] = ONE + ALPHAMINUS1_DT*matrix[1][1];

    const FLUCS_COMPLEX L00 = ONE + ALPHA_DT*matrix[0][0];
    const FLUCS_COMPLEX L01 = ALPHA_DT*matrix[0][1];
    const FLUCS_COMPLEX L10 = ALPHA_DT*matrix[1][0];
    const FLUCS_COMPLEX L11 = ONE + ALPHA_DT*matrix[1][1];

    const FLUCS_COMPLEX inv_det_L = ONE / (L00*L11 - L01*L10);

    inverse_lhs[0][0] = L11 * inv_det_L;
    inverse_lhs[0][1] = -L01 * inv_det_L;
    inverse_lhs[1][0] = -L10 * inv_det_L;
    inverse_lhs[1][1] = L00 * inv_det_L;
#elif NUMBER_OF_FIELDS == 3
    // TODO: hard-code 3 fields
#else
    // TODO: something clever
    // LU decomposition?
#endif
}


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

    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX inverse_lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_iteration_matrices(index, dt, rhs, inverse_lhs);

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            rhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = rhs[i][j];
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

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        rhs_fields[i] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            rhs_fields[i] += rhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * previous_fields[index + j*HALFUNPADDEDSIZE];
        }
    }

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_step, AB0, AB1, AB2, dft_bits, rhs_fields);
#endif

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        current_fields[index + i*HALFUNPADDEDSIZE] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            current_fields[index + i*HALFUNPADDEDSIZE] += inverse_lhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * rhs_fields[j];
        }
    }
#else // not PRECOMPUTE_LINEAR_MATRIX

    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX inverse_lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_iteration_matrices(index, dt, rhs, inverse_lhs);

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        rhs_fields[i] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            rhs_fields[i] += rhs[i][j] * previous_fields[index + j*HALFUNPADDEDSIZE];
        }
    }

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_step, AB0, AB1, AB2, dft_bits, rhs_fields);
#endif

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        current_fields[index + i*HALFUNPADDEDSIZE] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            current_fields[index + i*HALFUNPADDEDSIZE] += inverse_lhs[i][j] * rhs_fields[j];
        }
    }

#endif // PRECOMPUTE_LINEAR_MATRIX
}

} // extern "C"
