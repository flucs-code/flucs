/*
 * Shu-Osher RK3 method with Pade-approximated exponential integrating factors
 * for the nonlinear terms and forcing with Pade approximation for the linear part.
 */
#pragma once

__device__ __forceinline__ void compute_propagator(
    const size_t index,
    const FLUCS_FLOAT propagator_dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
){
    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    get_linear_matrix_wrapped(index, propagator_dt, current_time, current_step, propagator_dt, matrix);
    pade_exponential(matrix, lhs, propagator);
    gaussian_elimination_inplace<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(lhs, propagator, propagator);
}

// Precomputed linear propagators stored in global memory
extern "C" {

__device__ FLUCS_COMPLEX propagator_half_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];
__device__ FLUCS_COMPLEX propagator_full_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];
__device__ FLUCS_COMPLEX propagator_minus_half_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];

// Precomputes the half-step, full-step, and negative half-step linear propagators.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    // Precomputing should not be used with time-dependent linear matrices.
    compute_propagator(index, (FLUCS_FLOAT)(dt/2.0), (FLUCS_FLOAT)0, 0, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_half_precomp_global[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_half_precomp_global[i][j][index] = propagator[i][j];
        }
    }

    compute_propagator(index, dt, (FLUCS_FLOAT)0, 0, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_full_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_full_precomp_global[i][j][index] = propagator[i][j];
        }
    }

    compute_propagator(index, (FLUCS_FLOAT)(-dt/2.0), (FLUCS_FLOAT)0, 0, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_minus_half_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_minus_half_precomp_global[i][j][index] = propagator[i][j];
        }
    }
}

} // extern "C"

// Gets the explicit terms for the stage rhs.
__device__ void get_explicit_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
    const FLUCS_COMPLEX previous_stage_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
)
{
    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        explicit_terms[i] = 0;
    }

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_time, current_step, dft_bits_global, explicit_terms);
#endif

#ifdef FORCING_EXPLICIT
    FLUCS_COMPLEX forcing_fields[NUMBER_OF_FIELDS];
    get_forcing_fields(
        index, previous_stage_global, forcing_fields
    );
    add_forcing_explicit(
        index, dt, current_time, current_step,
        forcing_fields, explicit_terms
    );
#endif
}

// Called right at the end of an SSPRK3 stage,
// combines the linear matrices and nonlinear
// terms to find the fields at the next stage.
template<int stage>
__global__ void finish_stage(
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_FLOAT adaptive_rate,
    const FLUCS_COMPLEX previous_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
    FLUCS_COMPLEX stage_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    FLUCS_COMPLEX current_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE]
){

    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX previous_fields[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX stage_fields[NUMBER_OF_FIELDS] = {0};
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS] = {0};
    FLUCS_COMPLEX result[NUMBER_OF_FIELDS] = {0};

    // Load previous_fields from global memory
    #pragma unroll
    for (int j = 0; j < NUMBER_OF_FIELDS; j++){
        previous_fields[j] = previous_fields_global[j][index];

        if constexpr (stage > 1) {
            stage_fields[j] = stage_fields_global[j][index];
        }
    }

    FLUCS_COMPLEX propagator_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_full[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_minus_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    const FLUCS_FLOAT half_dt = dt / (FLUCS_FLOAT)2.0;

#ifdef PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            if constexpr (stage == 2 || stage == 3) {
                propagator_half[i][j] = propagator_half_precomp_global[i][j][index];
            }
            if constexpr (stage == 1 || stage == 3) {
                propagator_full[i][j] = propagator_full_precomp_global[i][j][index];
            }
            if constexpr (stage == 2) {
                propagator_minus_half[i][j] = propagator_minus_half_precomp_global[i][j][index];
            }
        }
    }

#else // not PRECOMPUTE_LINEAR_MATRIX

    if constexpr (stage == 1) {
        // Full propagator t_n -> t_n + dt
        compute_propagator(index, dt, current_time, current_step, propagator_full);
    }

    if constexpr (stage == 2) {
        // Half propagator t_n -> t_n + dt/2
        compute_propagator(index, half_dt, current_time, current_step, propagator_half);
        // Negative half propagator t_n + dt -> t_n + dt/2 
        compute_propagator(index, -half_dt, current_time + dt, current_step, propagator_minus_half);
    }

    if constexpr (stage == 3) {
        // Full propagator t_n -> t_n + dt
        compute_propagator(index, dt, current_time, current_step, propagator_full);
        // Half propagator t_n + dt/2 -> t_n + dt
        compute_propagator(index, half_dt, current_time + half_dt, current_step, propagator_half);
    }

#endif // PRECOMPUTE_LINEAR_MATRIX

    // Stage 1
    if constexpr (stage == 1) {

        get_explicit_terms(
            index, dt, current_time, current_step, dft_bits_global, previous_fields_global, explicit_terms
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += propagator_full[i][j] * (previous_fields[j] - dt*explicit_terms[j]);
            }

            result[i] = sum;
        }

        complete_timestep_stage(
            index, dt, current_time + dt, current_step, result
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            stage_fields_global[i][index] = result[i];
        }
    }

    // Stage 2
    else if constexpr (stage == 2) {

        get_explicit_terms(
            index, dt, current_time + dt, current_step, dft_bits_global, stage_fields_global, explicit_terms
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += ((FLUCS_FLOAT)(3.0/4.0)) * propagator_half[i][j]       * previous_fields[j] \
                      +((FLUCS_FLOAT)(1.0/4.0)) * propagator_minus_half[i][j] * (stage_fields[j] - dt*explicit_terms[j]);
            }

            result[i] = sum;
        }

        complete_timestep_stage(
            index, dt, current_time + half_dt, current_step, result
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            stage_fields_global[i][index] = result[i];
        }
    }

    // Stage 3
    else if constexpr (stage == 3) {

        get_explicit_terms(
            index, dt, current_time + half_dt, current_step, dft_bits_global, stage_fields_global, explicit_terms
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
                sum += ((FLUCS_FLOAT)(1.0/3.0)) * propagator_full[i][j] * previous_fields[j] \
                      +((FLUCS_FLOAT)(2.0/3.0)) * propagator_half[i][j] * (stage_fields[j] - dt*explicit_terms[j]);
#else
                sum += propagator_full[i][j]*previous_fields[j];
#endif
            }

            result[i] = sum;
        }

        complete_timestep_stage(
            index, dt, current_time + dt, current_step, result
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields_global[i][index] = result[i];
        }

        complete_finish_step(index, dt, current_time + dt, current_step, adaptive_rate, current_fields_global);
    }
}
