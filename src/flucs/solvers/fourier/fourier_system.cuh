#pragma once

#include <cupy/complex.cuh>

// Deal with float types
#ifdef DOUBLE_PRECISION
    #define FLUCS_FLOAT double
    #define flucs_fabs(x) fabs(x)
    #define flucs_sqrt(x) sqrt(x)
    #define flucs_fmax(x, y) fmax(x, y)
    #define FLUCS_COMPLEX_FLOAT_EQUIV double2
    #define FLUCS_EPSILON ((FLUCS_FLOAT)2.2204460492503131e-16)
#else
    #define FLUCS_FLOAT float
    #define flucs_fabs(x) fabsf(x)
    #define flucs_sqrt(x) sqrtf(x)
    #define flucs_fmax(x, y) fmaxf(x, y)
    #define FLUCS_COMPLEX_FLOAT_EQUIV float2
    #define FLUCS_EPSILON ((FLUCS_FLOAT)1.1920928955078125e-7f)
#endif

#define FLUCS_COMPLEX complex<FLUCS_FLOAT>

#define FLOAT_ONE ((FLUCS_FLOAT)1.0)
#define COMPLEX_ONE FLUCS_COMPLEX(FLOAT_ONE, 0)

// Includes 
#include "flucs/solvers/fourier/cuda/utilities.cuh"
#include "flucs/solvers/fourier/cuda/indexing.cuh"
#include "flucs/solvers/fourier/cuda/reductions.cuh"
#include "flucs/solvers/fourier/cuda/hyperdissipation.cuh"
#include "flucs/solvers/fourier/cuda/linear_pade.cuh"
#ifdef FORCING
#include "flucs/solvers/fourier/cuda/forcing.cuh"
#endif

extern "C" {

// Gets the linear matrix for a single mode.
// Must be implemented by the user.
__device__ void get_linear_matrix(const size_t index,
                                  const FLUCS_FLOAT dt,
                                  const FLUCS_FLOAT current_time,
                                  const long long current_step,
                                  FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]);

// Finds the nonlinear terms for the current time.
// Must be implemented by the user.
__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]);

// Forcing terms
#ifdef FORCING

#ifndef FORCING_FROM_SOLVER

#ifdef FORCING_EXPLICIT
__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]);
#endif

#ifdef FORCING_LINEAR
__device__ void add_forcing_linear(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]);
#endif

#endif // not FORCING_FROM_SOLVER
#endif // FORCING

// Wrapper for get_linear_matrix that adds an overall scaling factor
// and forcing (if needed)
__device__ __forceinline__
void get_linear_matrix_wrapped(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_FLOAT scale,
    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {

    get_linear_matrix(index, dt, current_time, current_step, matrix);

#ifdef FORCING_LINEAR
    add_forcing_linear(index, dt, current_time, current_step, matrix);
#endif

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            matrix[i][j] *= scale;
        }
    }
}

// Returns the linear matrix in canonical-grid storage, with padded modes zero.
// Matrix is assumed to be contiguous with shape
// (NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, index).
__global__ void compute_linear_matrix(
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX linear_matrix_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFSIZE])
{
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < HALFSIZE))
        return;

    if (is_mode_padded(index)) {
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
                linear_matrix_global[i][j][index] = 0;
            }
        }
        return;
    }

    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_linear_matrix_wrapped(index, dt, current_time, current_step, FLOAT_ONE, matrix);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            linear_matrix_global[i][j][index] = matrix[i][j];
        }
    }
}
// Adds hyperdissipation to the final fields
__device__ __forceinline__
void add_hyperdissipation(
    const size_t index,
    const FLUCS_FLOAT propagator_dt,
    const FLUCS_FLOAT adaptive_rate,
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {
    #if !(defined(HYPERDISSIPATION_KPERP) || defined(HYPERDISSIPATION_KX) || \
          defined(HYPERDISSIPATION_KY)    || defined(HYPERDISSIPATION_KZ))
        return;
    #endif

    const FLUCS_FLOAT hyperdissipation = (
        get_hyperdissipation(index, adaptive_rate)
    );
    const FLUCS_FLOAT factor = exp(-propagator_dt * hyperdissipation);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            propagator[i][j] *= factor;
        }
    }
}

// Function for end-of-timestep-stage operations (e.g., divergence cleaning)
#ifdef COMPLETE_TIMESTEP_STAGE
__device__ void complete_timestep_stage(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX fields[NUMBER_OF_FIELDS]
);
#else // no-op if not implemented
__device__ __forceinline__ void complete_timestep_stage(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX fields[NUMBER_OF_FIELDS]
) {}
#endif

// Function for end-of-timestep operations
__device__ __forceinline__
void complete_finish_step(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX current_fields_global[NUMBER_OF_FIELDS][HALFSIZE]
) {
    ;
}

} // extern "C"

template<bool include_hyperdissipation = true>
__device__ __forceinline__ void compute_propagator(
    const size_t index,
    const FLUCS_FLOAT propagator_dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_FLOAT adaptive_rate,
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
){
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    get_linear_matrix_wrapped(index, propagator_dt, current_time, current_step, propagator_dt, matrix);
    pade_exponential(matrix, lhs, propagator);
    gaussian_elimination_inplace<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(lhs, propagator, propagator);

    if constexpr (include_hyperdissipation) {
        add_hyperdissipation(index, propagator_dt, adaptive_rate, propagator);
    }
}


extern "C" {

// Returns one for solved modes and zero for dealiasing-only modes.
__global__ void compute_solved_grid_mask(
    FLUCS_FLOAT solved_grid_mask_global[HALFSIZE]
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < HALFSIZE))
        return;

    solved_grid_mask_global[index] = (
        is_mode_padded(index) ? (FLUCS_FLOAT)0.0 : FLOAT_ONE
    );
}

// Returns the full-grid linear propagator without hyperdissipation, with
// padded modes zero.
__global__ void compute_propagator_global(
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX propagator_global
        [NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFSIZE]
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < HALFSIZE))
        return;

    if (is_mode_padded(index)) {
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
                propagator_global[i][j][index] = 0;
            }
        }
        return;
    }

    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    compute_propagator<false>(
        index,
        dt,
        current_time,
        current_step,
        FLOAT_ONE,
        propagator
    );

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            propagator_global[i][j][index] = propagator[i][j];
        }
    }
}

} // extern "C"


#ifdef AB3
#include "flucs/solvers/fourier/timesteppers/ab3.cuh"
#endif

#ifdef RK4
#include "flucs/solvers/fourier/timesteppers/rk4.cuh"
#endif

#ifdef SSPRK3
#include "flucs/solvers/fourier/timesteppers/ssprk3.cuh"
#endif
