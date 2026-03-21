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

// Convering between padded and unpadded
// Care must be taken not to overflow size_t!!!
__device__ __forceinline__
size_t ikx_from_padded_ikx(const size_t padded_ikx) {
    return  (padded_ikx < HALF_NX) ? padded_ikx : (NX + padded_ikx) - PADDED_NX ;
}

__device__ __forceinline__
size_t ikz_from_padded_ikz(const size_t padded_ikz) {
    return  (padded_ikz < HALF_NZ) ? padded_ikz : (NZ + padded_ikz) - PADDED_NZ ;
}

// We have implicity assumed that PADDED_N_ > N_
__device__ __forceinline__
size_t padded_ikx_from_ikx(const size_t ikx) {
    return  (ikx < HALF_NX) ? ikx : (PADDED_NX - NX) + ikx;
}

__device__ __forceinline__
size_t padded_ikz_from_ikz(const size_t ikz) {
    return  (ikz < HALF_NZ) ? ikz : (PADDED_NZ - NZ) + ikz;
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
    union {size_t ikx, padded_ikx, ix;};
    union {size_t iky, padded_iky, iy;};
    union {size_t ikz, padded_ikz, iz;};
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
