/*
 * AB3 method for the nonlinear terms and forcing
 * with Pade approximation for the linear part.
 */
#pragma once

extern "C" {
// Multistep explicit terms stored in global memory
__constant__ FLUCS_COMPLEX* multistep_explicit_terms;

// Precomputed matrices stored in global memory
__constant__ FLUCS_COMPLEX* rhs_precomp = NULL;
__constant__ FLUCS_COMPLEX* inverse_lhs_precomp = NULL;

// Precomputes the rhs and inverse_lhs matrices.
__global__ void precompute_iteration_matrices(const FLUCS_FLOAT dt){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    // Precomputing should not be used with time-dependent linear matrices
    get_linear_matrix_wrapped(index, dt, (FLUCS_FLOAT)0, 0, dt, matrix);

    pade_lhs_rhs(matrix, lhs, rhs);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            rhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] =\
                rhs[i][j];
        }
    }

    FLUCS_COMPLEX inverse_lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
    invert_matrix_inplace(lhs, inverse_lhs);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            inverse_lhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] = inverse_lhs[i][j];
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
    FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS]
)
{
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS] = {0};

#ifdef NONLINEAR
    add_nonlinear_terms(index, dt, current_time, current_step, dft_bits, explicit_terms);
#endif

#ifdef FORCING_EXPLICIT
    add_forcing_explicit(index, dt, current_time, current_step, previous_fields, explicit_terms);
#endif

    const size_t multistep_index_0 = ((current_step      % 3 + 3) % 3) * NUMBER_OF_FIELDS * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_1 = ((current_step + 2) % 3)          * NUMBER_OF_FIELDS * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_2 = ((current_step + 1) % 3)          * NUMBER_OF_FIELDS * HALFUNPADDEDSIZE + index;

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        const size_t offset = i * HALFUNPADDEDSIZE;

        rhs_fields[i] -= dt * (
            + AB0 * explicit_terms[i]
            + AB1 * multistep_explicit_terms[multistep_index_1 + offset]
            + AB2 * multistep_explicit_terms[multistep_index_2 + offset]
        );

        multistep_explicit_terms[multistep_index_0 + offset] = explicit_terms[i];
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

    FLUCS_COMPLEX rhs_fields[NUMBER_OF_FIELDS];
    FLUCS_COMPLEX result[NUMBER_OF_FIELDS];

#ifdef PRECOMPUTE_LINEAR_MATRIX

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            sum += rhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * prev[j];
        }
        rhs_fields[i] = sum;
    }

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_time, current_step, AB0, AB1, AB2, dft_bits, previous_fields, rhs_fields);
#endif

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        FLUCS_COMPLEX sum = 0;

        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++){
            sum += inverse_lhs_precomp[index + HALFUNPADDEDSIZE*(j + NUMBER_OF_FIELDS*i)] * rhs_fields[j];
        }
        result[i] = sum;
    }
#else // not PRECOMPUTE_LINEAR_MATRIX

    FLUCS_COMPLEX lhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

    // Help the compiler a bit with the registers
    {
        FLUCS_COMPLEX rhs[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];
        FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS];

        get_linear_matrix_wrapped(index, dt, current_time, current_step, dt, matrix);
        pade_lhs_rhs(matrix, lhs, rhs);

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_FIELDS; i++){
            FLUCS_COMPLEX sum = 0;

            #pragma unroll
            for (int j = 0; j < NUMBER_OF_FIELDS; j++){
                sum += rhs[i][j] * prev[j];
            }

            rhs_fields[i] = sum;
        }
    }

#if defined(NONLINEAR) || defined(FORCING_EXPLICIT)
    add_explicit_terms(index, dt, current_time, current_step, AB0, AB1, AB2, dft_bits, previous_fields, rhs_fields);
#endif

    gaussian_elimination_inplace(lhs, result, rhs_fields);

#endif // PRECOMPUTE_LINEAR_MATRIX

    complete_timestep_stage(index, dt, current_time, current_step, result);

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++){
        current_fields[index + i*HALFUNPADDEDSIZE] = result[i];
    }

    complete_finish_step(
        index, dt, current_time, current_step, adaptive_rate, current_fields
    );

}

} // extern "C"
