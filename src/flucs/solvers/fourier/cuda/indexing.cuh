// Contains a bunch of useful indexing functions
#pragma once

// Wavenumbers from indices
__device__ __forceinline__
FLUCS_FLOAT kx_from_ikx(size_t ikx) {
    return (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : -TWOPI_OVER_LX * (NX - ikx);
}

__device__ __forceinline__
FLUCS_FLOAT ky_from_iky(size_t iky) {
    return TWOPI_OVER_LY * iky;
}

__device__ __forceinline__
FLUCS_FLOAT kz_from_ikz(size_t ikz) {
    return  (ikz < HALF_NZ) ? TWOPI_OVER_LZ * ikz : -TWOPI_OVER_LZ * (NZ - ikz);
}

// Fourier-space derivatives from indices
__device__ __forceinline__
FLUCS_COMPLEX dx_from_ikx(size_t ikx) {
    return FLUCS_COMPLEX(0, kx_from_ikx(ikx));
}

__device__ __forceinline__
FLUCS_COMPLEX dy_from_iky(size_t iky) {
    return FLUCS_COMPLEX(0, ky_from_iky(iky));
}

__device__ __forceinline__
FLUCS_COMPLEX dz_from_ikz(size_t ikz) {
    return FLUCS_COMPLEX(0, kz_from_ikz(ikz));
}

__device__ __forceinline__
FLUCS_COMPLEX get_phase_shift_factor(size_t ikz, size_t ikx, size_t iky) {

    constexpr FLUCS_FLOAT PI_OVER_NX = FLUCS_PI / (FLUCS_FLOAT)NX;
    constexpr FLUCS_FLOAT PI_OVER_NY = FLUCS_PI / (FLUCS_FLOAT)NY;
    constexpr FLUCS_FLOAT PI_OVER_NZ = FLUCS_PI / (FLUCS_FLOAT)NZ;

    const FLUCS_FLOAT phase = (
          ( (ikx < HALF_NX) ? PI_OVER_NX * ikx : -PI_OVER_NX * (NX - ikx) )
        + ( (ikz < HALF_NZ) ? PI_OVER_NZ * ikz : -PI_OVER_NZ * (NZ - ikz) )
        + (FLUCS_FLOAT)(iky) * PI_OVER_NY
    );

    return FLUCS_COMPLEX(flucs_cos(phase), flucs_sin(phase));
}


// Converting between 3D and linear indexing
// nz is not used but it is retained for signature consistency
template<size_t nz, size_t nx, size_t ny>
__device__ __forceinline__
size_t index_from_3d(const size_t ikz, const size_t ikx, const size_t iky) {
    return iky + ny * (ikx + nx * ikz);
}

struct indices3d_t {
    union {size_t ikx, ix;};
    union {size_t iky, iy;};
    union {size_t ikz, iz;};
};

// Given a linear index in a 3D array of shape (nz, nx, ny)
// find the corresponding 3D index (iz, ix, iy)
// nz is not used but it is retained for signature consistency
template<size_t nz, size_t nx, size_t ny>
__device__ __forceinline__
indices3d_t get_indices3d(const size_t index) {
    indices3d_t result;

    const size_t intermediate = index / ny;
    result.iy = index - intermediate * ny;

    result.iz = intermediate / nx;
    result.ix = intermediate - result.iz * nx;
    return result;
}

// Check whether a mode is padded given specific indices
__device__ __forceinline__
bool is_mode_padded(const size_t ikz, const size_t ikx, const size_t iky) {

#ifdef TWO_THIRDS_DEALIASING
    return (   (ikx >= HALF_NX_UNPADDED && ikx < (HALF_NX_UNPADDED + NX) - NX_UNPADDED)
            || (ikz >= HALF_NZ_UNPADDED && ikz < (HALF_NZ_UNPADDED + NZ) - NZ_UNPADDED)
            ||  iky >= HALF_NY_UNPADDED);
#endif

#ifdef PHASE_SHIFT_DEALIASING
    constexpr FLUCS_FLOAT ONE_OVER_NZ = FLOAT_ONE / (FLUCS_FLOAT)NZ;
    constexpr FLUCS_FLOAT ONE_OVER_NX = FLOAT_ONE / (FLUCS_FLOAT)NX;
    constexpr FLUCS_FLOAT ONE_OVER_NY = FLOAT_ONE / (FLUCS_FLOAT)NY;

    // DFT wavenumbers
    const FLUCS_FLOAT qz_abs = (FLUCS_FLOAT)( (ikz < HALF_NZ) ? ikz : NZ - ikz ) * ONE_OVER_NZ;
    const FLUCS_FLOAT qx_abs = (FLUCS_FLOAT)( (ikx < HALF_NX) ? ikx : NX - ikx ) * ONE_OVER_NX;
    const FLUCS_FLOAT qy_abs = (FLUCS_FLOAT)(iky) * ONE_OVER_NY;

#ifdef PHASE_SHIFT_MAXIMAL
    // Get rid of Nyquist

    if constexpr (NZ % 2 == 0) {
        if (ikz == NZ / 2)
            return true;
    }
    if constexpr (NX % 2 == 0) {
        if (ikx == NX / 2)
            return true;
    }
    if constexpr (NY % 2 == 0) {
        if (iky == NY / 2)
            return true;
    }

    // Anything above max_sum will needs to be dealiased
    constexpr FLUCS_FLOAT max_sum = 0.666;
    return (
           (qx_abs + qz_abs > max_sum)
        || (qy_abs + qz_abs > max_sum)
        || (qx_abs + qy_abs > max_sum)
    );
#endif // PHASE_SHIFT_MAXIMAL

#ifdef PHASE_SHIFT_SPHERICAL
    
    return (
        qx_abs*qx_abs + qy_abs*qy_abs + qz_abs*qz_abs
    ) > DEALIASING_RADIUS_SQUARED;

#endif // PHASE_SHIFT_SPHERICAL

#endif // PHASE_SHIFT_DEALIASING
}

// Check whether a mode is padded given a linear index
__device__ __forceinline__
bool is_mode_padded(const size_t index) {
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    return is_mode_padded(ikz, ikx, iky);
}
