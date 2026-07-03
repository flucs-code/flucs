/*
 * Classic RK4 method with Pade-approximated exponential integrating factors for
 * the nonlinear terms and forcing with Pade approximation for the linear part.
 */
#pragma once

// Precomputed linear propagator stored in global memory
__constant__ FLUCS_COMPLEX* propagator_half_precomp = NULL;
__constant__ FLUCS_COMPLEX* propagator_full_precomp = NULL;

// // Precomputes the rhs and inverse_lhs matrices.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;


    // Not implemented
    __trap();
    return;

    // // Check if we are within bounds
    // if (!(index < HALFUNPADDEDSIZE))
    //     return;
    //
    // // This will first holds rhs then the propagator = lhs^-1 rhs
    // FLUCS_COMPLEX propagator[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    //
    // { // Help those registers
    //     FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    //
    //     FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    //
    //     // Precomputing should not be used with time-dependent linear matrices
    //     get_linear_matrix_wrapped(index, dt, (FLUCS_FLOAT)0, 0, dt, matrix);
    //
    //     pade_lhs_rhs(matrix, lhs, propagator);
    //
    //     // In-place gaussian elimination
    //     gaussian_elimination_inplace<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(lhs, propagator, propagator);
    // }
    //
    // #pragma unroll
    // for (int i = 0; i < NUMBER_OF_FIELDS; i++){
    //     #pragma unroll
    //     for (int j = 0; j < NUMBER_OF_FIELDS; j++){
    //         propagator_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] =\
    //             propagator[i][j];
    //     }
    // }
}

// Gets the explicit terms for the stage rhs and the current_field update
template<int stage>
__device__ void get_explicit_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX* dft_bits,
    const FLUCS_COMPLEX* previous_fields,
    FLUCS_COMPLEX stage_fields[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX current_fields[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX propagator_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS],
    FLUCS_COMPLEX propagator_full[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
)
{
    // Explicitly treated terms calculated at the previous time step
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS] = {0};

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_time, current_step, dft_bits, explicit_terms);
#endif

#ifdef FORCING_EXPLICIT
    add_forcing_explicit(index, dt, current_time, current_step, previous_fields, explicit_terms);
#endif

    FLUCS_FLOAT stage_weight, current_weight;

    if constexpr (stage == 1) {
        stage_weight = (FLUCS_FLOAT)(-dt/2.0);
        current_weight = (FLUCS_FLOAT)(-dt/6.0);
    }
    else if constexpr (stage == 2) {
        stage_weight = (FLUCS_FLOAT)(-dt/2.0);
        current_weight = (FLUCS_FLOAT)(-dt/3.0);
    }
    else if constexpr (stage == 3) {
        stage_weight = (FLUCS_FLOAT)(-dt);
        current_weight = (FLUCS_FLOAT)(-dt/3.0);
    }
    else if constexpr (stage == 4) {
        current_weight = (FLUCS_FLOAT)(-dt/6.0);
    }

    // First, multiply the explicit terms by the propagator
    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){

        FLUCS_COMPLEX stage_sum = 0;
        FLUCS_COMPLEX current_sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){

            // stage field update
            if constexpr (stage == 1 || stage == 3) {
                stage_sum += propagator_half[i][j] * explicit_terms[j];
            }

            // current field update
            if constexpr (stage == 1) {
                current_sum += propagator_full[i][j] * explicit_terms[j];
            }
            else if constexpr (stage < 4) {
                current_sum += propagator_half[i][j] * explicit_terms[j];
            }
        }

        if constexpr (stage == 1 || stage == 3) {
            stage_fields[i] += stage_weight * stage_sum;
        }
        else if constexpr (stage == 2) {
            stage_fields[i] += stage_weight * explicit_terms[i];
        }

        if constexpr (stage < 4) {
            current_fields[i] += current_weight * current_sum;
        }
        else {
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
    const FLUCS_FLOAT adaptive_rate,
    const FLUCS_COMPLEX* previous_fields_global,
    const FLUCS_COMPLEX* dft_bits,
    FLUCS_COMPLEX* stage_fields_global,
    FLUCS_COMPLEX* current_fields_global
){

    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;


    FLUCS_COMPLEX previous_fields[NUMBER_OF_FIELDS];
    // Load previous_fields from global memory
    #pragma unroll
    for (int j = 0; j < NUMBER_OF_FIELDS; j++){
        previous_fields[j] = previous_fields_global[index + j*HALFUNPADDEDSIZE];
    }

    FLUCS_COMPLEX stage_fields[NUMBER_OF_FIELDS] = {0};
    FLUCS_COMPLEX current_fields[NUMBER_OF_FIELDS] = {0};
    FLUCS_COMPLEX propagator_half[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX propagator_full[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

#ifdef PRECOMPUTE_LINEAR_MATRIX

#error "not implemented"
    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            propagator_half[i][j] = propagator_half_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)];
            propagator_full[i][j] = propagator_full_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)];

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

        // Half propagator needed if stage < 4
        if constexpr (stage < 4) {
            get_linear_matrix_wrapped(index, (FLUCS_FLOAT)(dt/2.0), current_time, current_step, (FLUCS_FLOAT)(dt/2.0), matrix);
            pade_lhs_rhs(matrix, lhs, propagator_half);

            // In-place gaussian elimination
            gaussian_elimination_inplace<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(lhs, propagator_half, propagator_half);
        }

        // Full propagator needed if stage == 1 or 3
        if constexpr (stage == 1 || stage == 3) {
            get_linear_matrix_wrapped(index, dt, current_time, current_step, dt, matrix);
            pade_lhs_rhs(matrix, lhs, propagator_full);

            // In-place gaussian elimination
            gaussian_elimination_inplace<NUMBER_OF_FIELDS, NUMBER_OF_FIELDS>(lhs, propagator_full, propagator_full);
        }
    }


#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    get_explicit_terms<stage>(index, dt, current_time, current_step, dft_bits, previous_fields, stage_fields, current_fields, propagator_half, propagator_full);
#endif

    // Stage update
    if constexpr (stage < 4) {
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                if constexpr (stage < 3) {
                    sum += propagator_half[i][j] * previous_fields[j];
                }
                else {
                    sum += propagator_full[i][j] * previous_fields[j];
                }
            }

            stage_fields[i] += sum;
        }
    }

    // Linear part for current_fields
    if constexpr (stage == 1) {
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += propagator_full[i][j] * previous_fields[j];
            }

            current_fields[i] += sum;
        }
    }


#endif // PRECOMPUTE_LINEAR_MATRIX

    // Store stage
    if constexpr (stage < 4) {
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            stage_fields_global[index + i*HALFUNPADDEDSIZE] = stage_fields[i];
        }
    }

    if constexpr (stage == 1) {
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields_global[index + i*HALFUNPADDEDSIZE] = current_fields[i];
        }
    }
    else {
#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            current_fields_global[index + i*HALFUNPADDEDSIZE] += current_fields[i];
        }
#endif
    }


    if constexpr (stage == 4) {
        complete_finish_step(
            index, dt, current_time, current_step, adaptive_rate, current_fields_global
        );
    }

}
