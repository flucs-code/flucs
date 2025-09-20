#pragma once

#include <cupy/complex.cuh>

// Deal with float types

#ifdef DOUBLE_PRECISION
    #define FLUCS_FLOAT double
    #define flucs_fabs(x) fabs(x)
#else
    #define FLUCS_FLOAT float
    #define flucs_fabs(x) fabsf(x)
#endif

#define FLUCS_COMPLEX complex<FLUCS_FLOAT>


extern "C" {

__constant__ FLUCS_COMPLEX* R_precomp = NULL;
__constant__ FLUCS_COMPLEX* invL_precomp = NULL;


// Gets the linear matrix for a single mode.
// This must be implemented by the user.
__device__ void get_linear_matrix(const int index,
                                  const FLUCS_FLOAT dt,
                                  FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]);

// Finds iterations matrices for a single mode.
__device__ void get_iteration_matrices(const int index,
                                       const FLUCS_FLOAT dt,
                                       FLUCS_COMPLEX R[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
                                       FLUCS_COMPLEX invL[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]){

    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    get_linear_matrix(index, dt, matrix);
#if NUMBER_OF_FIELDS == 1
    // TODO: one field
#elif NUMBER_OF_FIELDS == 2
    // Hard-coded 2x2 inversion
    const FLUCS_FLOAT ALPHA_DT = ALPHA*dt;
    const FLUCS_FLOAT ALPHAMINUS1_DT = (ALPHA - 1)*dt;
    const FLUCS_FLOAT ONE = (FLUCS_FLOAT)1.0;

    R[0][0] = ONE + ALPHAMINUS1_DT*matrix[0][0];
    R[0][1] = ALPHAMINUS1_DT*matrix[0][1];
    R[1][0] = ALPHAMINUS1_DT*matrix[1][0];
    R[1][1] = ONE + ALPHAMINUS1_DT*matrix[1][1];

    const FLUCS_COMPLEX L00 = ONE + ALPHA_DT*matrix[0][0];
    const FLUCS_COMPLEX L01 = ALPHA_DT*matrix[0][1];
    const FLUCS_COMPLEX L10 = ALPHA_DT*matrix[1][0];
    const FLUCS_COMPLEX L11 = ONE + ALPHA_DT*matrix[1][1];

    const FLUCS_COMPLEX inv_det_L = ONE / (L00*L11 - L01*L10);

    invL[0][0] = L11 * inv_det_L;
    invL[0][1] = -L01 * inv_det_L;
    invL[1][0] = -L10 * inv_det_L;
    invL[1][1] = L00 * inv_det_L;
#elif NUMBER_OF_FIELDS == 3
    // TODO: hard-code 3 fields
#else
    // TODO
    // LU decomposition?
#endif
}


// Returns the full (for all modes) linear matrix
// matrix is assumed to be contiguous with shape (NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, index)
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


// Precomputes the R and invL matrices.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX R[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX invL[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_iteration_matrices(index, dt, R, invL);

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            R_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = R[i][j];
            invL_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = invL[i][j];
        }
    }

}


__global__ void finish_step(const FLUCS_COMPLEX* fields,
                   FLUCS_COMPLEX* result,
                   const FLUCS_FLOAT dt) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS];

#ifdef PRECOMPUTE_LINEAR_MATRIX

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        rhs_fields[i] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            rhs_fields[i] += R_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * fields[index + j*HALFUNPADDEDSIZE];
        }
    }

    // Nonlinear terms added here

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        result[index + i*HALFUNPADDEDSIZE] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            result[index + i*HALFUNPADDEDSIZE] += invL_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * rhs_fields[j];
        }
    }
#else

    FLUCS_COMPLEX R[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX invL[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_iteration_matrices(index, dt, R, invL);

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        rhs_fields[i] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            rhs_fields[i] += R[i][j] * fields[index + j*HALFUNPADDEDSIZE];
        }
    }

    // Nonlinear terms added here

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        result[index + i*HALFUNPADDEDSIZE] = 0;

        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            result[index + i*HALFUNPADDEDSIZE] += invL[i][j] * rhs_fields[j];
        }
    }

#endif
}

} // extern "C"
