/* CUDA implementation of the three-dimensional Fourier test system. */

#include "flucs/solvers/fourier/fourier_system.cuh"

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Core solver functions
////////////////////////////////////////////////////////////////////////////////

// Linear terms
__device__ void get_linear_matrix(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
) {
    // Indices
    const indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    // Wavenumbers
    const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
    const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
    const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

    // Advection frequency
    const FLUCS_FLOAT advection_frequency =
        + ADVECTION_X * kx
        + ADVECTION_Y * ky
        + ADVECTION_Z * kz;

    #pragma unroll
    for (int i = 0; i < NUMBER_OF_FIELDS; i++) {
        #pragma unroll
        for (int j = 0; j < NUMBER_OF_FIELDS; j++) {
            matrix[i][j] = 0;
        }
        matrix[i][i] = FLUCS_COMPLEX(0, advection_frequency);
    }

    // Omega cross u in field order (uz, ux, uy)
    matrix[0][1] = -ROTATION_Y;
    matrix[0][2] = +ROTATION_X;
    matrix[1][0] = +ROTATION_Y;
    matrix[1][2] = -ROTATION_Z;
    matrix[2][0] = -ROTATION_X;
    matrix[2][1] = +ROTATION_Z;
}

// Fourier fields for the pseudospectral nonlinear calculation
__global__ void find_derivatives(
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX dft_derivatives_global
        [NUMBER_OF_DFT_DERIVATIVES][HALFSIZE]
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (!(index < HALFSIZE))
        return;

    #pragma unroll
    for (int field = 0; field < NUMBER_OF_FIELDS; field++) {
        dft_derivatives_global[field][index] = is_mode_padded(index)
            ? FLUCS_COMPLEX(0, 0)
            : fields_global[field][index];
    }
}

// Real-space products for the nonlinear terms
__global__ void find_nonlinear_bits(
    const FLUCS_FLOAT real_derivatives_global
        [NUMBER_OF_DFT_DERIVATIVES][FULLSIZE],
    FLUCS_FLOAT real_bits_global[NUMBER_OF_DFT_BITS][FULLSIZE],
    const bool calculate_cfl,
    FLUCS_FLOAT* cfl_rate_global
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = index < FULLSIZE;

    // Fields follow the Fourier-axis order (uz, ux, uy).
    const FLUCS_FLOAT uz = in_bounds
        ? real_derivatives_global[0][index]
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT ux = in_bounds
        ? real_derivatives_global[1][index]
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT uy = in_bounds
        ? real_derivatives_global[2][index]
        : (FLUCS_FLOAT)0;

    // CFL rate based on maximum velocity in each direction
    if (calculate_cfl) {
        const FLUCS_FLOAT cfl_rate =
            + (flucs_fabs(uz) + flucs_fabs(ux)) * (NZ_UNPADDED / LZ)
            + (flucs_fabs(ux) + flucs_fabs(uy)) * (NX_UNPADDED / LX)
            + (flucs_fabs(uy) + flucs_fabs(uz)) * (NY_UNPADDED / LY);
        update_cfl(cfl_rate, cfl_rate_global);
    }

    if (!in_bounds)
        return;

    // Read all three fields before writing: inputs and outputs may alias.
    // Bits are grouped by nonlinear component in field order (z, x, y).
    real_bits_global[0][index] = ((FLUCS_FLOAT)0.5) * uy * uy;
    real_bits_global[1][index] = ux * uz;
    real_bits_global[2][index] = ux * uy;
    real_bits_global[3][index] = ((FLUCS_FLOAT)0.5) * uz * uz;
    real_bits_global[4][index] = ((FLUCS_FLOAT)0.5) * ux * ux;
    real_bits_global[5][index] = uy * uz;
}

__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global
        [NUMBER_OF_DFT_BITS][HALFSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
) {
    // Indices
    const indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    // Derivatives
    const FLUCS_COMPLEX dx = dx_from_ikx(indices.ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(indices.iky);
    const FLUCS_COMPLEX dz = dz_from_ikz(indices.ikz);

    // Calculate nonlinear terms
    explicit_terms[0] += DFT_FULLSIZE_FACTOR * (
        + dy * dft_bits_global[0][index]
        + dz * dft_bits_global[1][index]
    );
    explicit_terms[1] += DFT_FULLSIZE_FACTOR * (
        + dx * dft_bits_global[2][index]
        + dz * dft_bits_global[3][index]
    );
    explicit_terms[2] += DFT_FULLSIZE_FACTOR * (
        + dx * dft_bits_global[4][index]
        + dy * dft_bits_global[5][index]
    );
}

////////////////////////////////////////////////////////////////////////////////
// Model helper functions
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ FLUCS_FLOAT get_free_energy_rate_from_explicit_terms(
    const size_t index,
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    const FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
) {
    FLUCS_FLOAT result = 0;
    #pragma unroll
    for (int field = 0; field < NUMBER_OF_FIELDS; field++) {
        
        // Field
        const FLUCS_COMPLEX value = fields_global[field][index];

        // Explicit terms evolution
        result -= + value.real() * explicit_terms[field].real()
                  + value.imag() * explicit_terms[field].imag();
    }
    return result;
}


////////////////////////////////////////////////////////////////////////////////
// Forcing
////////////////////////////////////////////////////////////////////////////////

// Forcing
#ifdef FORCING_METHOD_NEGATIVE_DAMPING
__device__ void add_forcing_explicit(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_forcing[NUMBER_OF_FIELDS],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
) {
    if (!forcing_range_mask(index))
        return;

    #pragma unroll
    for (int field = 0; field < NUMBER_OF_FIELDS; field++) {
        // Explicit terms are stored on the left-hand side of the equation.
        explicit_terms[field] -= FORCING_RATE * previous_fields_forcing[field];
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Diagnostics: Free Energy
////////////////////////////////////////////////////////////////////////////////

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        FLUCS_FLOAT result = 0;
        #pragma unroll
        for (int field = 0; field < NUMBER_OF_FIELDS; field++) {

            // Get the relevant field
            const FLUCS_COMPLEX value = fields_global[index + field * HALFSIZE];

            // Mod-square
            result += value.real() * value.real() + value.imag() * value.imag();
        }
        return ((FLUCS_FLOAT)0.5) * result;
    }
};

struct FreeEnergyNonlinear_Functor {
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        FLUCS_COMPLEX nonlinear_terms[NUMBER_OF_FIELDS] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(
            index,
            dt,
            current_time,
            current_step,
            dft_bits_global,
            nonlinear_terms
        );
#endif
        return get_free_energy_rate_from_explicit_terms(
            index, fields_global, nonlinear_terms
        );
    }
};

struct FreeEnergyForcing_Functor {
    const FLUCS_COMPLEX (* __restrict__ fields_global)[HALFSIZE];
    const FLUCS_FLOAT dt;
    const FLUCS_FLOAT current_time;
    const long long current_step;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        FLUCS_COMPLEX forcing_terms[NUMBER_OF_FIELDS] = {0};
#ifdef FORCING_EXPLICIT
        FLUCS_COMPLEX fields_forcing[NUMBER_OF_FIELDS];
        get_forcing_fields(index, fields_global, fields_forcing);
        add_forcing_explicit(
            index,
            dt,
            current_time,
            current_step,
            fields_forcing,
            forcing_terms
        );
#endif
        return get_free_energy_rate_from_explicit_terms(
            index, fields_global, forcing_terms
        );
    }
};

struct FreeEnergyHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

} // extern "C"
