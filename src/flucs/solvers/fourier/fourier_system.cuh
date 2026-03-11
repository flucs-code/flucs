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

// Calculates the perpendicular hyperviscosity for a given kx, ky mode
__device__ __forceinline__
FLUCS_FLOAT get_hypervisc_perp(const FLUCS_FLOAT kx, const FLUCS_FLOAT ky) {

#ifdef HYPERVISC_PERP

    const FLUCS_FLOAT kperp2 = kx * kx + ky * ky;
    FLUCS_FLOAT hypervisc = HYPERVISC_PERP;

    #pragma unroll
    for (int i = 0; i < HYPERVISC_PERP_POWER; i++)
        hypervisc *= kperp2;

    return hypervisc;

#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the parallel hyperviscosity for a given kz mode
__device__ __forceinline__
FLUCS_FLOAT get_hypervisc_par(const FLUCS_FLOAT kz) {

#ifdef HYPERVISC_PAR

    const FLUCS_FLOAT kz2 = kz * kz;
    FLUCS_FLOAT hypervisc = HYPERVISC_PAR;

    #pragma unroll
    for (int i = 0; i < HYPERVISC_PAR_POWER; i++)
        hypervisc *= kz2;

    return hypervisc;
#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the hyperviscosity for a given mode
__device__ __forceinline__
FLUCS_FLOAT get_hypervisc(const int index) {

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
    const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
    const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

    return get_hypervisc_perp(kx, ky) + get_hypervisc_par(kz);
}

// Functor for calculating the size of the term due to perpendicular hyperviscosity for a given mode
struct HyperviscPerp_Functor {
    const FLUCS_COMPLEX* __restrict__ field;
    __device__ __forceinline__ FLUCS_FLOAT operator()(int index) const {
        
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        const FLUCS_FLOAT hypervisc = get_hypervisc_perp(kx, ky);

        return hypervisc * Abs2_Functor{field, FLOAT_ONE}(index);
    }
};

// Functor for calculating the size of the term due to parallel hyperviscosity for a given mode
struct HyperviscPar_Functor {
    const FLUCS_COMPLEX* __restrict__ field;
    __device__ __forceinline__ FLUCS_FLOAT operator()(int index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

        const FLUCS_FLOAT hypervisc = get_hypervisc_par(kz);

        return hypervisc * Abs2_Functor{field, FLOAT_ONE}(index);
    }
};


// Functor for calculating the total (parallel + perpendicular)
// hyperviscosity for a given mode
struct Hypervisc_Functor {
    const FLUCS_COMPLEX* __restrict__ field;
    __device__ __forceinline__ FLUCS_FLOAT operator()(int index) const {

        const FLUCS_FLOAT hypervisc = get_hypervisc(index);
        return hypervisc * Abs2_Functor{field, FLOAT_ONE}(index);
    }
};

extern "C" {

// calculation of hyperviscous losses for a given field
__global__
void hypervisc_perp_magnitude(const FLUCS_COMPLEX* field, FLUCS_FLOAT* output) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE, output, HyperviscPerp_Functor{field});
}

__global__
void hypervisc_par_magnitude(const FLUCS_COMPLEX* field, FLUCS_FLOAT* output) {
    add_and_sum_last_axis<HALF_NY, true>(
        FLOAT_ONE, output, HyperviscPar_Functor{field});
}


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
                                    FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS]);

// Wrapper for get_linear_matrix that adds hyperviscosity if needed
__device__ __forceinline__
void get_linear_matrix_wrapped(const int index,
                       const FLUCS_FLOAT dt,
                       FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]) {

    get_linear_matrix(index, dt, matrix);

#if !(defined(HYPERVISC_PERP) || defined(HYPERVISC_PAR))
    return;
#endif

    const FLUCS_FLOAT hypervisc = get_hypervisc(index);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++)
        matrix[i][i] += hypervisc;
    
}


// Returns the full (for all modes) linear matrix.
// Matrix is assumed to be contiguous with shape (NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, index)
__global__ void compute_linear_matrix(const FLUCS_FLOAT dt, FLUCS_COMPLEX* linear_matrix){
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_linear_matrix_wrapped(index, dt, matrix);

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
        get_linear_matrix_wrapped(index, dt, matrix);

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
        get_linear_matrix_wrapped(index, dt, matrix);

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
