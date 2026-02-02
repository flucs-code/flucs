// Contains a bunch of useful indexing functions
#pragma once

// Wavenumbers from indices
__device__ __forceinline__
FLUCS_FLOAT kx_from_ikx(int ikx) {
    return (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
}

__device__ __forceinline__
FLUCS_FLOAT ky_from_iky(int iky) {
    return TWOPI_OVER_LY * iky;
}

__device__ __forceinline__
FLUCS_FLOAT kz_from_ikz(int ikz) {
    return  (ikz < HALF_NZ) ? TWOPI_OVER_LZ * ikz : TWOPI_OVER_LZ * (ikz - NZ);
}

// Convering between padded and unpadded
__device__ __forceinline__
int ikx_from_padded_ikx(const int padded_ikx) {
    return  (padded_ikx < HALF_NX) ? padded_ikx : NX - PADDED_NX + padded_ikx;
}

__device__ __forceinline__
int ikz_from_padded_ikz(const int padded_ikz) {
    return  (padded_ikz < HALF_NZ) ? padded_ikz : NZ - PADDED_NZ + padded_ikz;
}

__device__ __forceinline__
int padded_ikx_from_ikx(const int ikx) {
    return  (ikx < HALF_NX) ? ikx : PADDED_NX - NX + ikx;
}

__device__ __forceinline__
int padded_ikz_from_ikz(const int ikz) {
    return  (ikz < HALF_NZ) ? ikz : PADDED_NZ - NZ + ikz;
}

// Converting between 3D and linear indexing
// nz is not used but I like it there for consistency
template<int nz, int nx, int ny>
__device__ __forceinline__
int index_from_3d(const int ikz, const int ikx, const int iky) {
    return iky + ny * (ikx + nx * ikz);
}

// __device__ __forceinline__
// int padded_index_from_3d(const int padded_ikx, const int padded_iky, const int padded_ikz) {
//     return padded_iky + HALF_PADDED_NY * (padded_ikx + PADDED_NX * padded_ikz);
// }

struct indices3d_t {
    union {int ikx, padded_ikx, ix;};
    union {int iky, padded_iky, iy;};
    union {int ikz, padded_ikz, iz;};
};

// Given a linear index in a 3D array of shape (nz, nx, ny)
// find the corresponding 3D index (iz, ix, iy)
// nz is not used but I like it there for consistency
template<int nz, int nx, int ny>
__device__ __forceinline__
indices3d_t get_indices3d(const int index) {
    indices3d_t result;

    const int intermediate = index / ny;
    result.iy = index - intermediate * ny;

    result.iz = intermediate / nx;
    result.ix = intermediate - result.iz * nx;
    return result;
}
