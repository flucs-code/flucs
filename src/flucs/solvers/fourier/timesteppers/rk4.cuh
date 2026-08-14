/*
 * Classic RK4 method with Pade-approximated exponential integrating factors for
 * the nonlinear terms and forcing with Pade approximation for the linear part.
 */
#pragma once

// Precomputed linear propagator stored in global memory
extern "C" {

__device__ FLUCS_COMPLEX propagator_half_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];
__device__ FLUCS_COMPLEX propagator_full_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];

// Precomputes the half-step and full-step linear propagators.
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
        index, one_over_two * dt, (FLUCS_FLOAT)0, 0,
        adaptive_rate, propagator
    );

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            propagator_half_precomp_global[i][j][index] \
            = propagator[i][j];
        }
    }

    compute_propagator(index, dt, (FLUCS_FLOAT)0, 0, adaptive_rate, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            propagator_full_precomp_global[i][j][index] \
            = propagator[i][j];
        }
    }
}

} // extern "C"

// Gets the explicit terms for the stage rhs and the current_field update
template<int stage>
__device__ void get_explicit_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
    const FLUCS_COMPLEX previous_stage_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    FLUCS_COMPLEX stage_fields[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX current_fields[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX propagator_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX propagator_full[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
)
{
    // Explicitly treated terms calculated at the previous time step
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS] = {0};

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

    FLUCS_FLOAT stage_weight, current_weight;

    // Stage 1
    if constexpr (stage == 1) {
        constexpr FLUCS_FLOAT one_over_two = (FLUCS_FLOAT)(1.0 / 2.0);
        constexpr FLUCS_FLOAT one_over_six = (FLUCS_FLOAT)(1.0 / 6.0);

        stage_weight   = -one_over_two * dt;
        current_weight = -one_over_six * dt;

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX stage_sum = 0;
            FLUCS_COMPLEX current_sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                stage_sum += propagator_half[i][j] * explicit_terms[j];
                current_sum += propagator_full[i][j] * explicit_terms[j];
            }

            stage_fields[i] += stage_weight * stage_sum;
            current_fields[i] += current_weight * current_sum;
        }
    }

    // Stage 2
    else if constexpr (stage == 2) {
        constexpr FLUCS_FLOAT one_over_two = (FLUCS_FLOAT)(1.0 / 2.0);
        constexpr FLUCS_FLOAT one_over_three = (FLUCS_FLOAT)(1.0 / 3.0);

        stage_weight   = -one_over_two * dt;
        current_weight = -one_over_three * dt;

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX current_sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                current_sum += propagator_half[i][j] * explicit_terms[j];
            }

            stage_fields[i] += stage_weight * explicit_terms[i];
            current_fields[i] += current_weight * current_sum;
        }
    }

    // Stage 3
    else if constexpr (stage == 3) {
        constexpr FLUCS_FLOAT one_over_three = (FLUCS_FLOAT)(1.0 / 3.0);

        stage_weight   = (FLUCS_FLOAT)(-dt    );
        current_weight = -one_over_three * dt;

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX stage_sum = 0;
            FLUCS_COMPLEX current_sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                stage_sum += propagator_half[i][j] * explicit_terms[j];
                current_sum += propagator_half[i][j] * explicit_terms[j];
            }

            stage_fields[i] += stage_weight * stage_sum;
            current_fields[i] += current_weight * current_sum;
        }
    }

    // Stage 4
    else if constexpr (stage == 4) {
        constexpr FLUCS_FLOAT one_over_six = (FLUCS_FLOAT)(1.0 / 6.0);

        current_weight = -one_over_six * dt;

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields[i] += current_weight * explicit_terms[i];
        }
    }
}

// Called right at the end of an RK4 stage,
// combines the linear matrices and nonlinear
// terms to find the fields at the current time step.
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
    // Load previous_fields from global memory
    #pragma unroll
    for (int j = 0; j < NUMBER_OF_FIELDS; j++){
        previous_fields[j] = previous_fields_global[j][index];
    }

    const FLUCS_FLOAT half_dt = one_over_two * dt;

    FLUCS_COMPLEX stage_fields[NUMBER_OF_FIELDS] = {0};
    FLUCS_COMPLEX current_fields[NUMBER_OF_FIELDS] = {0};
    FLUCS_COMPLEX propagator_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_full[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    // Propagator from t_n + dt/2 -> t_n + dt
    // Used only when dynamically computing the propagator at each time step
    // Otherwise, fall back to propagator_half
#ifdef PRECOMPUTE_LINEAR_MATRIX
    auto& propagator_half_half = propagator_half;
#else
    FLUCS_COMPLEX propagator_half_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
#endif

#ifdef PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            if constexpr (stage < 4) {
                propagator_half[i][j] = propagator_half_precomp_global[i][j][index];
            }
            if constexpr (stage == 1 || stage == 3) {
                propagator_full[i][j] = propagator_full_precomp_global[i][j][index];
            }
        }
    }

#else // not PRECOMPUTE_LINEAR_MATRIX
    
    const FLUCS_FLOAT adaptive_rate = FLOAT_ONE / dt;
    
    if constexpr (stage == 1) {
        // t_n -> t_n + dt/2
        compute_propagator(index, half_dt, current_time, current_step, adaptive_rate, propagator_half);
        // t_n -> t_n + dt
        compute_propagator(index, dt, current_time, current_step, adaptive_rate, propagator_full);
    }

    if constexpr (stage == 2) {
        // t_n -> t_n + dt/2
        compute_propagator(index, half_dt, current_time, current_step, adaptive_rate, propagator_half);
        // t_n + dt/2 -> t_n + dt
        compute_propagator(index, half_dt, current_time + half_dt, current_step, adaptive_rate, propagator_half_half);
    }
    if constexpr (stage == 3) {
        // t_n + dt/2 -> t_n + dt
        compute_propagator(index, half_dt, current_time + half_dt, current_step, adaptive_rate, propagator_half_half);
        // t_n -> t_n + dt
        compute_propagator(index, dt, current_time, current_step, adaptive_rate, propagator_full);
    }

    // // Half propagator needed if stage < 4
    // if constexpr (stage < 4) {
    //     compute_propagator(index, (FLUCS_FLOAT)(dt/2.0), current_time, current_step, propagator_half);
    // }
    //
    // // Full propagator needed if stage == 1 or 3
    // if constexpr (stage == 1 || stage == 3) {
    //     compute_propagator(index, dt, current_time, current_step, propagator_full);
    // }


#endif // PRECOMPUTE_LINEAR_MATRIX

    // Stage 1
    if constexpr (stage == 1) {

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
        get_explicit_terms<stage>(
            index, dt, current_time, current_step, dft_bits_global, previous_fields_global,
            stage_fields, current_fields, propagator_half, propagator_full
        );
#endif

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += propagator_half[i][j] * previous_fields[j];
            }

            stage_fields[i] += sum;
        }

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += propagator_full[i][j] * previous_fields[j];
            }

            current_fields[i] += sum;
        }

        complete_timestep_stage(
            index, dt, current_time + half_dt, current_step, stage_fields
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_stage_fields_global[i][index] = stage_fields[i];
        }

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields_global[i][index] = current_fields[i];
        }
    }

    // Stage 2
    else if constexpr (stage == 2) {
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
        get_explicit_terms<stage>(
            index, dt, current_time + half_dt, current_step, dft_bits_global, previous_stage_fields_global,
            stage_fields, current_fields, propagator_half_half, propagator_full
        );
#endif
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += propagator_half[i][j] * previous_fields[j];
            }

            stage_fields[i] += sum;
        }

        complete_timestep_stage(
            index, dt, current_time + half_dt, current_step, stage_fields
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_stage_fields_global[i][index] = stage_fields[i];
        }

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields_global[i][index] += current_fields[i];
        }
#endif
    }

    // Stage 3
    else if constexpr (stage == 3) {
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
        get_explicit_terms<stage>(
            index, dt, current_time + half_dt, current_step, dft_bits_global, previous_stage_fields_global,
            stage_fields, current_fields, propagator_half_half, propagator_full
        );
#endif
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += propagator_full[i][j] * previous_fields[j];
            }

            stage_fields[i] += sum;
        }

        complete_timestep_stage(
            index, dt, current_time + dt, current_step, stage_fields
        );

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_stage_fields_global[i][index] = stage_fields[i];
        }

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields_global[i][index] += current_fields[i];
        }
#endif
    }

    // Stage 4
    else if constexpr (stage == 4) {
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
        get_explicit_terms<stage>(
            index, dt, current_time + dt, current_step, dft_bits_global, previous_stage_fields_global,
            stage_fields, current_fields, propagator_half, propagator_full
        );
#endif
        FLUCS_COMPLEX result[NUMBER_OF_FIELDS];

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
            result[i] = current_fields_global[i][index] + current_fields[i];
#else
            result[i] = current_fields_global[i][index];
#endif
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
