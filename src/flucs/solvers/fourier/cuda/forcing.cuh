#pragma once

__device__ __forceinline__
bool forcing_range_mask(const size_t index)
{
#if defined(FORCING_KPERP2_MIN) && defined(FORCING_KPERP2_MAX) && \
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
    return true;
#endif
}

// Loads the physical, conjugate-symmetric field values used by forcing.
__device__ __forceinline__
void get_forcing_fields(
    const size_t index,
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
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

#ifdef FORCING_FROM_SOLVER
#error "FORCING_FROM_SOLVER is defined, but no shared Fourier forcing methods are implemented yet."
#endif
