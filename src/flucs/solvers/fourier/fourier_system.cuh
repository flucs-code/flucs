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
    const FLUCS_COMPLEX* dft_bits,
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]);

// Forcing terms
#ifdef FORCING

#ifdef FORCING_FROM_SOLVER // Add forcing from shared methods
#include "flucs/solvers/fourier/fourier_system_forcing.cuh"
#else

#ifdef FORCING_EXPLICIT
__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX* previous_fields,
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

#endif // else FORCING_FROM_SOLVER
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

// Returns the full (for all modes) linear matrix.
// Matrix is assumed to be contiguous with shape (NUMBER_OF_FIELDS, NUMBER_OF_FIELDS, index)
__global__ void compute_linear_matrix(const FLUCS_FLOAT dt,
                                      const FLUCS_FLOAT current_time,
                                      const long long current_step,
                                      FLUCS_COMPLEX* linear_matrix) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    get_linear_matrix_wrapped(index, dt, current_time, current_step, FLOAT_ONE, matrix);

    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            linear_matrix[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = matrix[i][j];
        }
    }
}
// Adds hyperdissipation to the final fields
__device__ __forceinline__
void add_hyperdissipation(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT adaptive_rate,
    FLUCS_COMPLEX* current_fields
) {
    #if !(defined(HYPERDISSIPATION_KPERP) || defined(HYPERDISSIPATION_KX) || \
          defined(HYPERDISSIPATION_KY)    || defined(HYPERDISSIPATION_KZ))
        return;
    #endif

    const FLUCS_FLOAT hyperdissipation = (
        get_hyperdissipation(index, adaptive_rate)
    );
    const FLUCS_FLOAT factor = exp(-dt * hyperdissipation);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        current_fields[index + i * HALFUNPADDEDSIZE] *= factor;
    }
}

// Function for end-of-timestep operations
__device__ __forceinline__
void complete_finish_step(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_FLOAT adaptive_rate,
    FLUCS_COMPLEX* current_fields
) {
    add_hyperdissipation(index, dt, adaptive_rate, current_fields);
}

} // extern "C"

#ifdef AB3
#include "flucs/solvers/fourier/timesteppers/ab3.cuh"
#endif

#ifdef AB3_IF
#include "flucs/solvers/fourier/timesteppers/ab3_if.cuh"
#endif

#ifdef RK4
#include "flucs/solvers/fourier/timesteppers/rk4.cuh"
#endif
