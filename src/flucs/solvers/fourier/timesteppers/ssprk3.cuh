/*
 * Shu-Osher RK3 method with Pade-approximated exponential integrating factors
 * for the nonlinear terms and forcing with Pade approximation for the linear part.
 */
#pragma once

// Precomputed linear propagators stored in global memory
extern "C" {

__device__ FLUCS_COMPLEX propagator_half_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];
__device__ FLUCS_COMPLEX propagator_full_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];
__device__ FLUCS_COMPLEX propagator_minus_half_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];

// Precomputes the half-step, full-step, and negative half-step linear propagators.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    constexpr FLUCS_FLOAT one_over_two = (FLUCS_FLOAT)(1.0 / 2.0);

    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    const FLUCS_FLOAT adaptive_rate = FLOAT_ONE / dt;

    // Precomputing should not be used with time-dependent linear matrices.
    compute_propagator(
        index, one_over_two * dt, 0, 0, adaptive_rate, propagator
    );

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_half_precomp_global[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_half_precomp_global[i][j][index] = propagator[i][j];
        }
    }

    compute_propagator(index, dt, 0, 0, adaptive_rate, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_full_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_full_precomp_global[i][j][index] = propagator[i][j];
        }
    }

    compute_propagator(
        index, -one_over_two * dt, 0, 0, adaptive_rate, propagator
    );

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
    const FLUCS_COMPLEX previous_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
    const FLUCS_COMPLEX previous_stage_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    FLUCS_COMPLEX current_stage_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    FLUCS_COMPLEX current_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE]
){
    constexpr FLUCS_FLOAT one_over_two = (FLUCS_FLOAT)(1.0 / 2.0);

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
            stage_fields[j] = previous_stage_fields_global[j][index];
        }
    }

    FLUCS_COMPLEX propagator_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_full[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_minus_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    const FLUCS_FLOAT half_dt = one_over_two * dt;

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
    const FLUCS_FLOAT adaptive_rate = FLOAT_ONE / dt;

    if constexpr (stage == 1) {
        // Full propagator t_n -> t_n + dt
        compute_propagator(index, dt, current_time, current_step, adaptive_rate, propagator_full);
    }

    if constexpr (stage == 2) {
        // Half propagator t_n -> t_n + dt/2
        compute_propagator(index, half_dt, current_time, current_step, adaptive_rate, propagator_half);
        // Negative half propagator t_n + dt -> t_n + dt/2 
        compute_propagator(index, -half_dt, current_time + dt, current_step, adaptive_rate, propagator_minus_half);
    }

    if constexpr (stage == 3) {
        // Full propagator t_n -> t_n + dt
        compute_propagator(index, dt, current_time, current_step, adaptive_rate, propagator_full);
        // Half propagator t_n + dt/2 -> t_n + dt
        compute_propagator(index, half_dt, current_time + half_dt, current_step, adaptive_rate, propagator_half);
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
            current_stage_fields_global[i][index] = result[i];
        }
    }

    // Stage 2
    else if constexpr (stage == 2) {
        constexpr FLUCS_FLOAT one_over_four = (FLUCS_FLOAT)(1.0 / 4.0);
        constexpr FLUCS_FLOAT three_over_four = (FLUCS_FLOAT)(3.0 / 4.0);

        get_explicit_terms(
            index, dt, current_time + dt, current_step, dft_bits_global, previous_stage_fields_global, explicit_terms
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += three_over_four * propagator_half[i][j] * previous_fields[j] \
                      + one_over_four * propagator_minus_half[i][j] \
                        * (stage_fields[j] - dt*explicit_terms[j]);
            }

            result[i] = sum;
        }

        complete_timestep_stage(
            index, dt, current_time + half_dt, current_step, result
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_stage_fields_global[i][index] = result[i];
        }
    }

    // Stage 3
    else if constexpr (stage == 3) {
        constexpr FLUCS_FLOAT one_over_three = (FLUCS_FLOAT)(1.0 / 3.0);
        constexpr FLUCS_FLOAT two_over_three = (FLUCS_FLOAT)(2.0 / 3.0);

        get_explicit_terms(
            index, dt, current_time + half_dt, current_step, dft_bits_global, previous_stage_fields_global, explicit_terms
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
                sum += one_over_three * propagator_full[i][j] * previous_fields[j] \
                      + two_over_three * propagator_half[i][j] \
                        * (stage_fields[j] - dt*explicit_terms[j]);
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

        complete_finish_step(index, dt, current_time + dt, current_step, current_fields_global);
    }
}
