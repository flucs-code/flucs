#pragma once

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
