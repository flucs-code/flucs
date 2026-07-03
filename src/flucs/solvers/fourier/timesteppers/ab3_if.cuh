/*
 * AB3 method with Pade-approximated exponential integrating factors for
 * the nonlinear terms and forcing with Pade approximation for the linear part.
 */
#pragma once

extern "C" {
// Multistep explicit terms stored in global memory
__constant__ FLUCS_COMPLEX* multistep_explicit_terms;

// Precomputed linear propagator stored in global memory
__constant__ FLUCS_COMPLEX* propagator_precomp = NULL;

// Precomputes the rhs and inverse_lhs matrices.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    // This will first holds rhs then the propagator = lhs^-1 rhs
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    { // Help those registers
        FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

        FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

        // Precomputing should not be used with time-dependent linear matrices
        get_linear_matrix_wrapped(index, dt, (FLUCS_FLOAT)0, 0, dt, matrix);

        pade_lhs_rhs(matrix, lhs, propagator);

        // In-place gaussian elimination
        gaussian_elimination_inplace<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(lhs, propagator, propagator);
    }

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            propagator_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] =\
                propagator[i][j];
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
    const FLUCS_COMPLEX* dft_bits,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX result[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
)
{
    // Explicitly treated terms calculated at the previous time step
    FLUCS_COMPLEX explicit_terms_0[NUMBER_OF_FIELDS] = {0};

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_time, current_step, dft_bits, explicit_terms_0);
#endif

#ifdef FORCING_EXPLICIT
    add_forcing_explicit(index, dt, current_time, current_step, previous_fields, explicit_terms_0);
#endif

    const size_t multistep_index_0 = ((current_step      % 3 + 3) % 3) * NUMBER_OF_FIELDS * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_1 = ((current_step + 2) % 3)          * NUMBER_OF_FIELDS * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_2 = ((current_step + 1) % 3)          * NUMBER_OF_FIELDS * HALFUNPADDEDSIZE + index;

    FLUCS_COMPLEX propagator_explicit_0[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_explicit_1[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_explicit_2[NUMBER_OF_FIELDS];

    // Load history from global memory
    FLUCS_COMPLEX explicit_terms_1[NUMBER_OF_FIELDS], explicit_terms_2[NUMBER_OF_FIELDS];


    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        const size_t offset = i * HALFUNPADDEDSIZE;
        explicit_terms_1[i] = multistep_explicit_terms[multistep_index_1 + offset]; 
        explicit_terms_2[i] = multistep_explicit_terms[multistep_index_2 + offset]; 
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
        const size_t offset = i * HALFUNPADDEDSIZE;

        result[i] -= dt * (
            + AB0 * propagator_explicit_0[i]
            + AB1 * propagator_explicit_1[i]
            + AB2 * propagator_explicit_2[i]
        );

        // Store the required terms in global memory
        multistep_explicit_terms[multistep_index_0 + offset] = propagator_explicit_0[i];
        multistep_explicit_terms[multistep_index_1 + offset] = propagator_explicit_1[i];
    }
}

// Called right at the end of a time step,
// combines the linear matrices and nonlinear
// terms to find the fields at the current time step.
__global__ void finish_step(
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_FLOAT adaptive_rate,
    const FLUCS_FLOAT AB0,
    const FLUCS_FLOAT AB1,
    const FLUCS_FLOAT AB2,
    const FLUCS_COMPLEX* previous_fields,
    const FLUCS_COMPLEX* dft_bits,
    FLUCS_COMPLEX* current_fields
){

    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;


    FLUCS_COMPLEX prev[NUMBER_OF_FIELDS];
    // Load previous_fields from global memory
    #pragma unroll
    for (int j = 0; j < NUMBER_OF_FIELDS; j++){
        prev[j] = previous_fields[index + j*HALFUNPADDEDSIZE];
    }

    FLUCS_COMPLEX result[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

#ifdef PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            propagator[i][j] = propagator_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)];
            sum += propagator[i][j] * prev[j];
        }
        result[i] = sum;
    }

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_time, current_step, AB0, AB1, AB2, dft_bits, previous_fields, result, propagator);
#endif

#else // not PRECOMPUTE_LINEAR_MATRIX

    // Help the compiler a bit with the registers
    {
        FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
        FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

        get_linear_matrix_wrapped(index, dt, current_time, current_step, dt, matrix);
        pade_lhs_rhs(matrix, lhs, propagator);

        // In-place gaussian elimination
        gaussian_elimination_inplace<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(lhs, propagator, propagator);
    }

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            sum += propagator[i][j] * prev[j];
        }

        result[i] = sum;
    }


#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_time, current_step, AB0, AB1, AB2, dft_bits, previous_fields, result, propagator);
#endif

#endif // PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        current_fields[index + i*HALFUNPADDEDSIZE] = result[i];
    }


    complete_finish_step(
        index, dt, current_time, current_step, adaptive_rate, current_fields
    );

}

} // extern "C"
