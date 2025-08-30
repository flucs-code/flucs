/*
 * Contains all the CUDA kernels for the 2D ITG model of Ivanov et al. (2020).
 */

// A lot of basic functionality is already implemented here.
// #include "../../solvers/fourier/fourier.cu"
#include <cupy/complex.cuh>

#ifdef DOUBLE_PRECISION
    #define FLUCS_FLOAT double
    #define flucs_fabs(x) fabs(x)
#else
    #define FLUCS_FLOAT float
    #define flucs_fabs(x) fabsf(x)
#endif

#define FLUCS_COMPLEX complex<FLUCS_FLOAT>

extern "C" __global__
void linear_kernel(const FLUCS_COMPLEX* fields,
                   FLUCS_COMPLEX* result,
                   const FLUCS_FLOAT dt) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(index < HALFUNPADDEDSIZE))
        return;

    // First, we need to figure out the kx and ky of the mode.
    const int ikx = index / HALF_NY;
    const int iky = index % HALF_NY;

    const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    const FLUCS_FLOAT ky = TWOPI_OVER_LY * iky;

    // if (ikx < HALF_NX) {
    //     kx = TWOPI_OVER_LX * ikx;
    //     // ikx_padded = ikx
    // } else {
    //     kx = TWOPI_OVER_LX * (ikx - NX);
    //     // ikx_padded_m = padded_nx + (ikx_m - nx)
    // }

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    const FLUCS_FLOAT eta_inv = (FLUCS_FLOAT)(1.0) / ((FLUCS_FLOAT)(iky > 0) + kperp2);


    // Generate the linear matrix
    const FLUCS_COMPLEX matrix_phiphi = FLUCS_COMPLEX(
        A_TIMES_CHI*kperp2*kperp2,
        -ky*(KAPPA_B - KAPPA_N) + KAPPA_T*kperp2*ky) * eta_inv;

    const FLUCS_COMPLEX matrix_phiT = FLUCS_COMPLEX(
        -B_TIMES_CHI*kperp2*kperp2,
        -ky*KAPPA_B) * eta_inv;

    const FLUCS_COMPLEX matrix_Tphi = FLUCS_COMPLEX(
        0,
        KAPPA_T*ky);

    const FLUCS_COMPLEX matrix_TT = FLUCS_COMPLEX(
        CHI*kperp2,
        0);

    const FLUCS_COMPLEX R_phiphi = (FLUCS_FLOAT)(1.0) + (ALPHA - 1)*dt*matrix_phiphi;
    const FLUCS_COMPLEX R_phiT = (ALPHA - 1)*dt*matrix_phiT;
    const FLUCS_COMPLEX R_Tphi = (ALPHA - 1)*dt*matrix_Tphi;
    const FLUCS_COMPLEX R_TT = (FLUCS_FLOAT)(1.0) + (ALPHA - 1)*dt*matrix_TT;

    const FLUCS_COMPLEX L_phiphi = (FLUCS_FLOAT)(1.0) + ALPHA*dt*matrix_phiphi;
    const FLUCS_COMPLEX L_phiT = ALPHA*dt*matrix_phiT;
    const FLUCS_COMPLEX L_Tphi = ALPHA*dt*matrix_Tphi;
    const FLUCS_COMPLEX L_TT = (FLUCS_FLOAT)(1.0) + ALPHA*dt*matrix_TT;

    const FLUCS_COMPLEX inv_det_L = (FLUCS_FLOAT)(1.0) / (L_phiphi*L_TT - L_phiT*L_Tphi);

    const FLUCS_COMPLEX invL_phiphi = L_TT * inv_det_L;
    const FLUCS_COMPLEX invL_phiT = L_phiT * inv_det_L;
    const FLUCS_COMPLEX invL_Tphi = -L_Tphi * inv_det_L;
    const FLUCS_COMPLEX invL_TT = L_phiphi * inv_det_L;


    // phi 
    const FLUCS_COMPLEX rhs_phi = R_phiphi * fields[index] + R_phiT * fields[index + HALFUNPADDEDSIZE];

    // T
    const FLUCS_COMPLEX rhs_T = R_Tphi * fields[index] + R_TT * fields[index + HALFUNPADDEDSIZE];

    result[index] = invL_phiphi * rhs_phi + invL_phiT * rhs_T;
    result[index + HALFUNPADDEDSIZE] = invL_Tphi * rhs_phi + invL_TT * rhs_T;
}

