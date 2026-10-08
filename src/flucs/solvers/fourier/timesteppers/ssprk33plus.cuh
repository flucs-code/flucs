/*
 * Isherwood-Grant-Gottlieb (2018) eSSPRK+(3,3) (type of RK3) method with Pade-approximated exponential integrating factors
 * for the nonlinear terms and forcing with Pade approximation for the linear part.
 */
#pragma once

// Precomputed linear propagators stored in global memory
extern "C" {

#ifdef PRECOMPUTE_LINEAR_MATRIX
__device__ FLUCS_COMPLEX propagator_one_third_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFSIZE];
__device__ FLUCS_COMPLEX propagator_two_third_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFSIZE];
__device__ FLUCS_COMPLEX propagator_full_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFSIZE];

// Precomputes the one-third-step, two-third-step, and full-step linear propagators.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    constexpr FLUCS_FLOAT one_over_three = (FLUCS_FLOAT)(1.0 / 3.0);
    constexpr FLUCS_FLOAT two_over_three = (FLUCS_FLOAT)(2.0 / 3.0);


    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFSIZE))
        return;

    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    const FLUCS_FLOAT adaptive_rate = FLOAT_ONE / dt;

    // Precomputing should not be used with time-dependent linear matrices.
    compute_propagator(
        index, one_over_three * dt, 0, 0, adaptive_rate, propagator
    );

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_half_precomp_global[index + HALFSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_one_third_precomp_global[i][j][index] = propagator[i][j];
        }
    }

    compute_propagator(
        index, two_over_three * dt, 0, 0, adaptive_rate, propagator
    );

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_half_precomp_global[index + HALFSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_two_third_precomp_global[i][j][index] = propagator[i][j];
        }
    }

    compute_propagator(index, dt, 0, 0, adaptive_rate, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            // propagator_full_precomp[index + HALFSIZE*(j + NUMBER_OF_FIELDS*i)] = propagator[i][j];
            propagator_full_precomp_global[i][j][index] = propagator[i][j];
        }
    }

}

#endif // PRECOMPUTE_LINEAR_MATRIX

} // extern "C"

// Gets the explicit terms for the stage rhs.
__device__ void get_explicit_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFSIZE],
    const FLUCS_COMPLEX previous_stage_global[NUMBER_OF_FIELDS][HALFSIZE],
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

// Called right at the end of an SSPRK33Plus stage,
// combines the linear matrices and nonlinear
// terms to find the fields at the next stage.
template<int stage>
__global__ void finish_stage(
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFSIZE],
    const FLUCS_COMPLEX previous_stage_fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX current_stage_fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX current_fields_global[NUMBER_OF_FIELDS][HALFSIZE]
){
    constexpr FLUCS_FLOAT one_over_three = (FLUCS_FLOAT)(1.0 / 3.0);
    constexpr FLUCS_FLOAT two_over_three = (FLUCS_FLOAT)(2.0 / 3.0);


    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if  (!(index < HALFSIZE))
        return;

    if  (is_mode_padded(index)) {

        if constexpr (stage == 3) {
            #pragma unroll
            for (int i = 0; i < NUMBER_OF_FIELDS; i++){
                current_fields_global[i][index] = 0;
            }
        }
        return;
    }

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

    FLUCS_COMPLEX propagator_one_third[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_two_third[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_full[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    const FLUCS_FLOAT one_third_dt = one_over_three * dt;
    const FLUCS_FLOAT two_third_dt = two_over_three * dt;


#ifdef PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            if constexpr (stage == 1 || stage == 2) {
                propagator_two_third[i][j] = propagator_two_third_precomp_global[i][j][index];
            }
            if constexpr (stage == 1) {
                propagator_full[i][j] = propagator_full_precomp_global[i][j][index];
            }
            if constexpr (stage == 3) {
                propagator_one_third[i][j] = propagator_one_third_precomp_global[i][j][index];
            }
        }
    }

#else // not PRECOMPUTE_LINEAR_MATRIX
    const FLUCS_FLOAT adaptive_rate = FLOAT_ONE / dt;

    if constexpr (stage == 1 || stage == 2) {
        // Two-thirds propagator t_n -> t_n + 2dt/3
        compute_propagator(index, two_third_dt, current_time, current_step, adaptive_rate, propagator_two_third);
    }

    if constexpr (stage == 1) {
        // Full propagator t_n -> t_n + dt
        compute_propagator(index, dt, current_time, current_step, adaptive_rate, propagator_full);
    }

    if constexpr (stage == 3) {
        // One-third propagator t_n + 2dt/3 -> t_n + dt
        compute_propagator(index, one_third_dt, current_time + two_third_dt, current_step, adaptive_rate, propagator_one_third);
    }

#endif // PRECOMPUTE_LINEAR_MATRIX

    // Stage 1
    if constexpr (stage == 1) {
        constexpr FLUCS_FLOAT thirtyseven_over_sixtyfour = (FLUCS_FLOAT)(37.0 / 64.0);
        constexpr FLUCS_FLOAT five_over_thirtytwo = (FLUCS_FLOAT)(5.0 / 32.0);

        get_explicit_terms(
            index, dt, current_time, current_step, dft_bits_global, previous_fields_global, explicit_terms
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += propagator_two_third[i][j] * (previous_fields[j] - two_over_three*dt*explicit_terms[j]);
            }

            result[i] = sum;
        }

        complete_timestep_stage(
            index, dt, current_time + two_third_dt, current_step, result
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_stage_fields_global[i][index] = result[i];
        }

        // Accumulate everything that depends on the previous fields
        // into current_fields, to be completed in stage 3:
        // (59/128 + 15/128) P_full u^n - (15/128)(4/3) dt P_full E_1
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
                sum += propagator_full[i][j] \
                       * (thirtyseven_over_sixtyfour * previous_fields[j] \
                          - five_over_thirtytwo * dt*explicit_terms[j]);
#else
                sum += propagator_full[i][j]*previous_fields[j];
#endif
            }

            current_fields_global[i][index] = sum;
        }
    }

    // Stage 2
    else if constexpr (stage == 2) {
        constexpr FLUCS_FLOAT four_over_three = (FLUCS_FLOAT)(4.0 / 3.0);

        get_explicit_terms(
            index, dt, current_time + two_third_dt, current_step, dft_bits_global, previous_stage_fields_global, explicit_terms
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += two_over_three * propagator_two_third[i][j] * previous_fields[j];
            }

            result[i] = sum + one_over_three * (stage_fields[i] - four_over_three * dt*explicit_terms[i]);
        }

        complete_timestep_stage(
            index, dt, current_time + two_third_dt, current_step, result
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_stage_fields_global[i][index] = result[i];
        }
    }

    // Stage 3
    else if constexpr (stage == 3) {
        constexpr FLUCS_FLOAT four_over_three = (FLUCS_FLOAT)(4.0 / 3.0);
        constexpr FLUCS_FLOAT twentyseven_over_sixtyfour = (FLUCS_FLOAT)(27.0 / 64.0);

        get_explicit_terms(
            index, dt, current_time + two_third_dt, current_step, dft_bits_global, previous_stage_fields_global, explicit_terms
        );

        // Start from the stage-1 contribution accumulated in current_fields
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = current_fields_global[i][index];

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += twentyseven_over_sixtyfour * propagator_one_third[i][j] \
                       * (stage_fields[j] - four_over_three * dt*explicit_terms[j]);
            }
#endif

            result[i] = sum;
        }

        complete_timestep_stage(
            index, dt, current_time + dt, current_step, result
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields_global[i][index] = result[i];
        }

        complete_finish_step(
            index, dt, current_time + dt, current_step, previous_fields_global, current_fields_global
        );
    }
}
