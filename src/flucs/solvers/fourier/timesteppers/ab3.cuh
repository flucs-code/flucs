/*
 * AB3 method with Pade-approximated exponential integrating factors for
 * the nonlinear terms and forcing with Pade approximation for the linear part.
 */
#pragma once

extern "C" {
// Multistep explicit terms stored in global memory
__device__ FLUCS_COMPLEX multistep_explicit_terms_global[3][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];

// Precomputed linear propagator stored in global memory
__device__ FLUCS_COMPLEX propagator_precomp_global[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS][HALFUNPADDEDSIZE];

// Precomputes the rhs and inverse_lhs matrices.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    // This will first holds rhs then the propagator = lhs^-1 rhs
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    const FLUCS_FLOAT adaptive_rate = FLOAT_ONE / dt;

    // Precomputing should not be used with time-dependent linear matrices.
    compute_propagator(index, dt, 0, 0, adaptive_rate, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            propagator_precomp_global[i][j][index] = propagator[i][j];
        }
    }
}

// Adds the explicit terms to the rhs and updates the AB3 history
__device__ void add_explicit_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_FLOAT AB0,
    const FLUCS_FLOAT AB1,
    const FLUCS_FLOAT AB2,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
    const FLUCS_COMPLEX previous_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    FLUCS_COMPLEX result[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
)
{
    // Explicitly treated terms calculated at the previous time step
    FLUCS_COMPLEX explicit_terms_0[NUMBER_OF_FIELDS] = {0};

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_time, current_step, dft_bits_global, explicit_terms_0);
#endif

#ifdef FORCING_EXPLICIT
    FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS];
    get_forcing_fields(
        index, previous_fields_global, previous_fields_forcing
    );
    add_forcing_explicit(
        index, dt, current_time, current_step,
        previous_fields_forcing, explicit_terms_0
    );
#endif

    const size_t multistep_index_0 = ((current_step      % 3 + 3) % 3);
    const size_t multistep_index_1 = ((current_step + 2) % 3);
    const size_t multistep_index_2 = ((current_step + 1) % 3);

    FLUCS_COMPLEX propagator_explicit_0[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_explicit_1[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_explicit_2[NUMBER_OF_FIELDS];

    // Load history from global memory
    FLUCS_COMPLEX explicit_terms_1[NUMBER_OF_FIELDS], explicit_terms_2[NUMBER_OF_FIELDS];


    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        explicit_terms_1[i] = multistep_explicit_terms_global[multistep_index_1][i][index];
        explicit_terms_2[i] = multistep_explicit_terms_global[multistep_index_2][i][index];
    }

    // First, multiply the explicit terms by the propagator
    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){

        FLUCS_COMPLEX sum_0 = 0;
        FLUCS_COMPLEX sum_1 = 0;
        FLUCS_COMPLEX sum_2 = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            FLUCS_COMPLEX prop = propagator[i][j];

            sum_0 += prop * explicit_terms_0[j];
            sum_1 += prop * explicit_terms_1[j];
            sum_2 += prop * explicit_terms_2[j];
        }

        propagator_explicit_0[i] = sum_0;
        propagator_explicit_1[i] = sum_1;
        propagator_explicit_2[i] = sum_2;
    }

    // Add contributions to the vector new fields
    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        result[i] -= dt * (
            + AB0 * propagator_explicit_0[i]
            + AB1 * propagator_explicit_1[i]
            + AB2 * propagator_explicit_2[i]
        );

        // Store the required terms in global memory
        multistep_explicit_terms_global[multistep_index_0][i][index] = propagator_explicit_0[i];
        multistep_explicit_terms_global[multistep_index_1][i][index] = propagator_explicit_1[i];
    }
}

// Called right at the end of a time step,
// combines the linear matrices and nonlinear
// terms to find the fields at the current time step.
__global__ void finish_step(
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_FLOAT AB0,
    const FLUCS_FLOAT AB1,
    const FLUCS_FLOAT AB2,
    const FLUCS_COMPLEX previous_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
    FLUCS_COMPLEX current_fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE]
){

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

    FLUCS_COMPLEX result[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

#ifdef PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            propagator[i][j] = propagator_precomp_global[i][j][index];
            sum += propagator[i][j] * previous_fields[j];
        }
        result[i] = sum;
    }

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_time, current_step, AB0, AB1, AB2, dft_bits_global, previous_fields_global, result, propagator);
#endif

#else // not PRECOMPUTE_LINEAR_MATRIX

    const FLUCS_FLOAT adaptive_rate = FLOAT_ONE / dt;

    compute_propagator(index, dt, current_time, current_step, adaptive_rate, propagator);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            sum += propagator[i][j] * previous_fields[j];
        }

        result[i] = sum;
    }


#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_time, current_step, AB0, AB1, AB2, dft_bits_global, previous_fields_global, result, propagator);
#endif

#endif // PRECOMPUTE_LINEAR_MATRIX

    complete_timestep_stage(index, dt, current_time, current_step, result);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        current_fields_global[i][index] = result[i];
    }


    complete_finish_step(
        index, dt, current_time, current_step, current_fields_global
    );

}

} // extern "C"
