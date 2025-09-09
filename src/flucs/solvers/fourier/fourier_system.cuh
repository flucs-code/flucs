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

// This must be implemented by the user
__device__ void get_linear_matrices(const int index,
                                    const FLUCS_FLOAT dt,
                                    FLUCS_COMPLEX R[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
                                    FLUCS_COMPLEX invL[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]);


__global__ void linear_kernel(const FLUCS_COMPLEX* fields,
                   FLUCS_COMPLEX* result,
                   const FLUCS_FLOAT dt) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX R[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX invL[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_linear_matrices(index, dt, R, invL);

    // Let's see if the 2D one below works first before doing this
    // FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS];
    //
    //
    // for (int i = 0; i < NUMBER_OF_FIELDS; i++){
    //     rhs_fields[i] = 0;
    //
    //     for (int j = 0; j < NUMBER_OF_FIELDS; j++){
    //         rhs_fields[i] += R[i][j] * fields[index + j*HALFUNPADDEDSIZE];
    //     }
    // }
    //
    // for (int i = 0; i < NUMBER_OF_FIELDS; i++){
    //     result[index + i*HALFUNPADDEDSIZE] = 0;
    //
    //     for (int j = 0; j < NUMBER_OF_FIELDS; j++){
    //         result[index + i*HALFUNPADDEDSIZE] += invL[i][j] * rhs_fields[j];
    //     }
    // }
    

    const FLUCS_COMPLEX rhs_phi = R[0][0] * fields[index] + R[0][1] * fields[index + HALFUNPADDEDSIZE];
    const FLUCS_COMPLEX rhs_T = R[1][0] * fields[index] + R[1][1] * fields[index + HALFUNPADDEDSIZE];

    result[index] = invL[0][0] * rhs_phi + invL[0][1] * rhs_T;
    result[index + HALFUNPADDEDSIZE] = invL[1][0] * rhs_phi + invL[1][1] * rhs_T;

}
}
