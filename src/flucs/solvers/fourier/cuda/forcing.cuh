#pragma once

////////////////////////////////////////////////////////////////////////////////
// Forcing helper functions
////////////////////////////////////////////////////////////////////////////////

// Mask for kz, kperp forcing
__device__ __forceinline__
bool forcing_range_mask_kzkperp(const size_t index)
{
#if defined(FORCING_RANGE_KZKPERP) && \
    defined(FORCING_KPERP2_MIN) && defined(FORCING_KPERP2_MAX) && \
    defined(FORCING_KZ_MIN) && defined(FORCING_KZ_MAX)

    // Indices
    const indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // Wavenumbers
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_FLOAT kz = kz_from_ikz(ikz);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    const FLUCS_FLOAT kz_abs = flucs_fabs(kz);

    return (
        kperp2 > FORCING_KPERP2_MIN &&
        kperp2 < FORCING_KPERP2_MAX &&
        kz_abs > FORCING_KZ_MIN &&
        kz_abs < FORCING_KZ_MAX
    );
#else
    (void)index;
    return false;
#endif
}

// Mask for isotropic kmod forcing
__device__ __forceinline__
bool forcing_range_mask_kmod(const size_t index)
{
#if defined(FORCING_RANGE_KMOD) && \
    defined(FORCING_KMOD2_MIN) && defined(FORCING_KMOD2_MAX)

    // Indices
    const indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // Wavenumbers
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_FLOAT kz = kz_from_ikz(ikz);

    const FLUCS_FLOAT kmod2 = kx*kx + ky*ky + kz*kz;

    return (
        kmod2 > FORCING_KMOD2_MIN &&
        kmod2 < FORCING_KMOD2_MAX
    );
#else
    (void)index;
    return false;
#endif
}

// Selecting appropriate forcing mask
__device__ __forceinline__
bool forcing_range_mask(const size_t index)
{
#ifdef FORCING_RANGE_KMOD
    return forcing_range_mask_kmod(index);
#elif defined(FORCING_RANGE_KZKPERP)
    return forcing_range_mask_kzkperp(index);
#else
    (void)index;
    return false;
#endif
}

// Loads the physical, conjugate-symmetric field values used by forcing.
__device__ __forceinline__
void get_forcing_fields(
    const size_t index,
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX fields_forcing[NUMBER_OF_FIELDS]
) {
    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        fields_forcing[i] = fields_global[i][index];
    }

    const indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    // Only the ky=0 modes are stored with their conjugate partners.
    if (indices.iky != 0)
        return;

    const size_t conjugate_ikz = indices.ikz == 0 ? 0 : NZ - indices.ikz;
    const size_t conjugate_ikx = indices.ikx == 0 ? 0 : NX - indices.ikx;
    const size_t conjugate_index = index_from_3d<NZ, NX, HALF_NY>(
        conjugate_ikz, conjugate_ikx, 0
    );

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        fields_forcing[i] = ((FLUCS_FLOAT)0.5) * (
            fields_forcing[i] + conj(fields_global[i][conjugate_index])
        );
    }
}

////////////////////////////////////////////////////////////////////////////////
// Forcing methods
////////////////////////////////////////////////////////////////////////////////

#ifdef FORCING_METHOD_ORNSTEIN_UHLENBECK

extern "C" {

__device__ FLUCS_COMPLEX ornstein_uhlenbeck_forcing_global
    [NUMBER_OF_FIELDS][HALFSIZE];

__global__ void update_ornstein_uhlenbeck_forcing(
    const FLUCS_FLOAT decay,
    const FLUCS_FLOAT innovation,
    const long long current_step,
    const unsigned long long rand_seed,
    const FLUCS_FLOAT amplitude_per_mode[NUMBER_OF_FIELDS]
) {
    // Define once
    constexpr FLUCS_FLOAT one_over_sqrt_two =
        (FLUCS_FLOAT)0.70710678118654752440;

    // Check that we are within bounds
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (!(index < HALFSIZE))
        return;

    // Return if we are not forcing
    if (is_mode_padded(index) || !forcing_range_mask(index))
        return;

    // Ensure that conjugate modes init. cuRAND with the same random squence.
    const indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    size_t canonical_index = index;
    bool take_conjugate = false;
    bool self_conjugate = false;

    if (indices.iky == 0) {
        const size_t conjugate_ikz   = indices.ikz == 0 ? 0 : NZ - indices.ikz;
        const size_t conjugate_ikx   = indices.ikx == 0 ? 0 : NX - indices.ikx;
        const size_t conjugate_index = index_from_3d<NZ, NX, HALF_NY>(
            conjugate_ikz, conjugate_ikx, 0
        );

        canonical_index = index < conjugate_index ? index : conjugate_index;
        take_conjugate  = index > conjugate_index;
        self_conjugate  = index == conjugate_index;
    }

    // Iterate over the fields
    #pragma unroll
    for (int field = 0; field < NUMBER_OF_FIELDS; field++) {
        // Each field and canonical mode uses an independent Philox sequence.
        curandStatePhilox4_32_10_t random_state;

        // Unique random sequence
        const unsigned long long sequence = (
              ((unsigned long long)field) * ((unsigned long long)HALFSIZE)
            + ((unsigned long long)canonical_index)
        );

        // Initialise cuRAND
        curand_init(
            rand_seed,
            sequence,
            (unsigned long long)(2 * current_step),
            &random_state
        );

        // Generate random values using the cuRAND normal distribution
        const FLUCS_COMPLEX_FLOAT_EQUIV random_values = (
            flucs_normal2(&random_state)
        );

        // Ensure that the random values are appropriately conjugates
        FLUCS_COMPLEX random;
        if (self_conjugate) {
            random = FLUCS_COMPLEX((FLUCS_FLOAT)random_values.x, 0);
        }
        else {
            const FLUCS_FLOAT imaginary_sign = take_conjugate
                ? - FLOAT_ONE
                : + FLOAT_ONE;
            random = one_over_sqrt_two * FLUCS_COMPLEX(
                (FLUCS_FLOAT)random_values.x,
                imaginary_sign * (FLUCS_FLOAT)random_values.y
            );
        }

        // Exact update of the Ornstein--Uhlenbeck process over one timestep.
        ornstein_uhlenbeck_forcing_global[field][index] = (
            + decay * ornstein_uhlenbeck_forcing_global[field][index]
            + innovation * amplitude_per_mode[field] * random
        );
    }
}

} // extern "C"

#endif // FORCING_METHOD_ORNSTEIN_UHLENBECK

#ifdef FORCING_FROM_SOLVER

__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
) {
    // Unused variables
    (void)dt;
    (void)current_time;
    (void)current_step;
    (void)previous_fields_forcing;

#ifdef FORCING_METHOD_ORNSTEIN_UHLENBECK
    // Return if the mode is not forced
    if (!forcing_range_mask(index))
        return;

    #pragma unroll
    for (int field = 0; field < NUMBER_OF_FIELDS; field++) {
        // Explicit terms are stored on the left-hand side of the equation.
        explicit_terms[field] -= (
            ornstein_uhlenbeck_forcing_global[field][index]
        );
    }
#endif
}

#endif // FORCING_FROM_SOLVER
