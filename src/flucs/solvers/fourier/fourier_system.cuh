#pragma once

#include <cupy/complex.cuh>

// Deal with float types
#ifdef DOUBLE_PRECISION
    #define FLUCS_FLOAT double
    #define flucs_fabs(x) fabs(x)
    #define flucs_fmax(x, y) fmax(x, y)
    #define FLUCS_COMPLEX_FLOAT_EQUIV double2
#else
    #define FLUCS_FLOAT float
    #define flucs_fabs(x) fabsf(x)
    #define flucs_fmax(x, y) fmaxf(x, y)
    #define FLUCS_COMPLEX_FLOAT_EQUIV float2
#endif

#define FLUCS_COMPLEX complex<FLUCS_FLOAT>

#define FLOAT_ONE ((FLUCS_FLOAT)1.0)
#define COMPLEX_ONE FLUCS_COMPLEX(FLOAT_ONE, 0)

// Includes 
#include "flucs/solvers/fourier/fourier_system_utilities.cuh"
#include "flucs/solvers/fourier/fourier_system_indexing.cuh"
#include "flucs/solvers/fourier/fourier_system_reductions.cuh"
#include "flucs/solvers/fourier/fourier_system_hyperdissipation.cuh"


extern "C" {

// Precomputed matrices stored in global memory
__constant__ FLUCS_COMPLEX* rhs_precomp = NULL;
__constant__ FLUCS_COMPLEX* inverse_lhs_precomp = NULL;

// Multistep explicit terms stored in global memory
__constant__ FLUCS_COMPLEX* multistep_explicit_terms;

// Gets the linear matrix for a single mode.
// Must be implemented by the user.
__device__ void get_linear_matrix(const size_t index,
                                  const FLUCS_FLOAT dt,
                                  FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]);

// Finds the nonlinear terms for the current time.
// Must be implemented by the user.
__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_COMPLEX* dft_bits,
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS_EXPLICIT]);

// Provides a mapping from the explicit term index 
// to the field index that it contributes to.
// Must be implemented by the user.
__device__ __forceinline__
int explicit_term_field_index(const int term_index);

// Forcing terms
#ifdef FORCING

#ifdef FORCING_FROM_SOLVER // Add forcing from shared methods
#include "flucs/solvers/fourier/fourier_system_forcing.cuh"
#else

#ifdef FORCING_EXPLICIT
__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt,
    const long long current_step,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS_EXPLICIT]);
#endif

#ifdef FORCING_LINEAR
__device__ void add_forcing_linear(
    const size_t index,
    const FLUCS_FLOAT dt,
    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]);
#endif

#endif // else FORCING_FROM_SOLVER
#endif // FORCING


// Adds the explicit terms to the rhs and updates the AB3 history
__device__ void add_explicit_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const long long current_step,
    const FLUCS_FLOAT AB0,
    const FLUCS_FLOAT AB1,
    const FLUCS_FLOAT AB2,
    const FLUCS_COMPLEX* dft_bits,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS]
)
{
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS_EXPLICIT] = {0};

#ifdef NONLINEAR
    add_nonlinear_terms(index, dft_bits, explicit_terms);
#endif

#ifdef FORCING_EXPLICIT
    add_forcing_explicit(index, dt, current_step, previous_fields, explicit_terms);
#endif

    const size_t multistep_index_0 = ((current_step      % 3 + 3) % 3) * NUMBER_OF_FIELDS_EXPLICIT * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_1 = ((current_step + 2) % 3)          * NUMBER_OF_FIELDS_EXPLICIT * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_2 = ((current_step + 1) % 3)          * NUMBER_OF_FIELDS_EXPLICIT * HALFUNPADDEDSIZE + index;

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS_EXPLICIT; i++) {
        const int field = explicit_term_field_index(i);
        const size_t offset = i * HALFUNPADDEDSIZE;

        rhs_fields[field] -= dt * (
            + AB0 * explicit_terms[i]
            + AB1 * multistep_explicit_terms[multistep_index_1 + offset]
            + AB2 * multistep_explicit_terms[multistep_index_2 + offset]
        );

        multistep_explicit_terms[multistep_index_0 + offset] = explicit_terms[i];
    }

}

// Wrapper for get_linear_matrix that adds hyperdissipation and forcing if needed
__device__ __forceinline__
void get_linear_matrix_wrapped(const size_t index,
                       const FLUCS_FLOAT dt,
                       FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]) {

    get_linear_matrix(index, dt, matrix);

#ifdef FORCING_LINEAR
    add_forcing_linear(index, dt, matrix);
#endif

#if !(defined(HYPERDISSIPATION_PERP) || defined(HYPERDISSIPATION_KX) ||\
      defined(HYPERDISSIPATION_KY) || defined(HYPERDISSIPATION_KZ))
    return;
#endif

    const FLUCS_FLOAT hyperdissipation = get_hyperdissipation(index, dt);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++)
        matrix[i][i] += hyperdissipation;
    
}


// Returns the full (for all modes) linear matrix.
// Matrix is assumed to be contiguous with shape (NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, index)
__global__ void compute_linear_matrix(const FLUCS_FLOAT dt, FLUCS_COMPLEX* linear_matrix){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

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
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

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
                            const long long current_step,
                            const FLUCS_FLOAT AB0,
                            const FLUCS_FLOAT AB1,
                            const FLUCS_FLOAT AB2,
                            const FLUCS_COMPLEX* previous_fields,
                            const FLUCS_COMPLEX* dft_bits,
                            FLUCS_COMPLEX* current_fields){

    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

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

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_step, AB0, AB1, AB2, dft_bits, previous_fields, rhs_fields);
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

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_step, AB0, AB1, AB2, dft_bits, previous_fields, rhs_fields);
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
