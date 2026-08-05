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



// Converting between 3D and linear indexing
// nz is not used but I like it there for consistency
template<size_t nz, size_t nx, size_t ny>
__device__ __forceinline__
size_t index_from_3d(const size_t ikz, const size_t ikx, const size_t iky) {
    return iky + ny * (ikx + nx * ikz);
}

// __device__ __forceinline__
// size_t padded_index_from_3d(const size_t padded_ikx, const size_t padded_iky, const size_t padded_ikz) {
//     return padded_iky + HALF_PADDED_NY * (padded_ikx + PADDED_NX * padded_ikz);
// }

struct indices3d_t {
    union {size_t ikx, ix;};
    union {size_t iky, iy;};
    union {size_t ikz, iz;};
};

// Given a linear index in a 3D array of shape (nz, nx, ny)
// find the corresponding 3D index (iz, ix, iy)
// nz is not used but I like it there for consistency
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
    return (   (ikx >= HALF_NX_UNPADDED && ikx < (HALF_NX_UNPADDED + NX) - NX_UNPADDED)
            || (ikz >= HALF_NZ_UNPADDED && ikz < (HALF_NZ_UNPADDED + NZ) - NZ_UNPADDED)
            || iky >= HALF_NY_UNPADDED);
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
