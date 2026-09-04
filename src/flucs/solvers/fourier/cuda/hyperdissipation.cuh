#pragma once

// Computes the maximum solved wavenumber for each hyperdissipation component.
// Runs once during CUDA initialisation.
extern "C" __global__
void compute_hyperdissipation_components_kmax(
    FLUCS_FLOAT hyperdissipation_components_kmax[4]
) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    // Initialise values
    FLUCS_FLOAT kz_max = 0;
    FLUCS_FLOAT kx_max = 0;
    FLUCS_FLOAT ky_max = 0;
    FLUCS_FLOAT kperp2_max = 0;

    // The maximum kx, ky, and kperp occur for kz = 0
    for (size_t ikx = 0; ikx < NX; ikx++) {
        for (size_t iky = 0; iky < HALF_NY; iky++) {
            if (is_mode_padded(0, ikx, iky))
                continue;
            
            // Wavenumbers
            const FLUCS_FLOAT kx = kx_from_ikx(ikx);
            const FLUCS_FLOAT ky = ky_from_iky(iky);
            const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

            kx_max = flucs_fmax(kx_max, flucs_fabs(kx));
            ky_max = flucs_fmax(ky_max, flucs_fabs(ky));
            kperp2_max = flucs_fmax(kperp2_max, kperp2);
        }
    }

    // The maximum kz occurs on the kx = ky = 0 axis.
    for (size_t ikz = 0; ikz < NZ; ikz++) {
        if (is_mode_padded(ikz, 0, 0))
            continue;

        kz_max = flucs_fmax(
            kz_max,
            flucs_fabs(kz_from_ikz(ikz))
        );
    }

    // Store maximum wavenumbers
    hyperdissipation_components_kmax[HYPERDISSIPATION_KZ_INT] = kz_max;
    hyperdissipation_components_kmax[HYPERDISSIPATION_KX_INT] = kx_max;
    hyperdissipation_components_kmax[HYPERDISSIPATION_KY_INT] = ky_max;
    hyperdissipation_components_kmax[HYPERDISSIPATION_KPERP_INT] = (
        flucs_sqrt(kperp2_max)
    );
}


// Calculates the perpendicular hyperdissipation for a given kx, ky mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_kperp(
    const FLUCS_FLOAT kx,
    const FLUCS_FLOAT ky,
    const FLUCS_FLOAT adaptive_rate
) {

#ifdef HYPERDISSIPATION_KPERP

    const FLUCS_FLOAT kperp2 = kx * kx + ky * ky;

#ifdef HYPERDISSIPATION_KPERP_NORMALISED
    constexpr FLUCS_FLOAT kperp_max = HYPERDISSIPATION_KPERP_KMAX;
    const FLUCS_FLOAT kperp2_norm = kperp_max > 0
        ? kperp2 / (kperp_max * kperp_max)
        : (FLUCS_FLOAT)0;
#else
    const FLUCS_FLOAT kperp2_norm = kperp2;
#endif // NORMALISED

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KPERP;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KPERP_POWER; i++)
        hyperdissipation *= kperp2_norm;

    #ifdef HYPERDISSIPATION_KPERP_ADAPTIVE
        hyperdissipation *= adaptive_rate;
    #endif

    return hyperdissipation;

#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the kx hyperdissipation for a given kx mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_kx(
    const FLUCS_FLOAT kx,
    const FLUCS_FLOAT adaptive_rate
) {

#ifdef HYPERDISSIPATION_KX

#ifdef HYPERDISSIPATION_KX_NORMALISED
    constexpr FLUCS_FLOAT kx_max = HYPERDISSIPATION_KX_KMAX;
    const FLUCS_FLOAT kx_norm = kx_max > 0
        ? kx / kx_max
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT kx2_norm = kx_norm * kx_norm;
#else
    const FLUCS_FLOAT kx2_norm = kx * kx;
#endif

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KX;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KX_POWER; i++)
        hyperdissipation *= kx2_norm;

    #ifdef HYPERDISSIPATION_KX_ADAPTIVE
        hyperdissipation *= adaptive_rate;
    #endif

    return hyperdissipation;
#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the ky hyperdissipation for a given ky mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_ky(
    const FLUCS_FLOAT ky,
    const FLUCS_FLOAT adaptive_rate
) {

#ifdef HYPERDISSIPATION_KY

#ifdef HYPERDISSIPATION_KY_NORMALISED
    constexpr FLUCS_FLOAT ky_max = HYPERDISSIPATION_KY_KMAX;
    const FLUCS_FLOAT ky_norm = ky_max > 0
        ? ky / ky_max
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT ky2_norm = ky_norm * ky_norm;
#else
    const FLUCS_FLOAT ky2_norm = ky * ky;
#endif

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KY;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KY_POWER; i++)
        hyperdissipation *= ky2_norm;

    #ifdef HYPERDISSIPATION_KY_ADAPTIVE
        hyperdissipation *= adaptive_rate;
    #endif

    return hyperdissipation;
#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the kz hyperdissipation for a given kz mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_kz(
    const FLUCS_FLOAT kz,
    const FLUCS_FLOAT adaptive_rate
) {

#ifdef HYPERDISSIPATION_KZ

#ifdef HYPERDISSIPATION_KZ_NORMALISED
    constexpr FLUCS_FLOAT kz_max = HYPERDISSIPATION_KZ_KMAX;
    const FLUCS_FLOAT kz_norm = kz_max > 0
        ? kz / kz_max
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT kz2_norm = kz_norm * kz_norm;
#else
    const FLUCS_FLOAT kz2_norm = kz * kz;
#endif

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KZ;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KZ_POWER; i++)
        hyperdissipation *= kz2_norm;

    #ifdef HYPERDISSIPATION_KZ_ADAPTIVE
        hyperdissipation *= adaptive_rate;
    #endif

    return hyperdissipation;
#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the total hyperdissipation for a given mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation(
    const size_t index,
    const FLUCS_FLOAT adaptive_rate
) {

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
    const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
    const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

    return get_hyperdissipation_kperp(kx, ky, adaptive_rate)
        + get_hyperdissipation_kx(kx, adaptive_rate)
        + get_hyperdissipation_ky(ky, adaptive_rate)
        + get_hyperdissipation_kz(kz, adaptive_rate);
}

// Functor for calculating the size of the term due to perpendicular hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKperp_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT adaptive_rate;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        const FLUCS_FLOAT hyperdissipation = (
            get_hyperdissipation_kperp(kx, ky, adaptive_rate)
        );

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to kx hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKx_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT adaptive_rate;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);

        const FLUCS_FLOAT hyperdissipation = (
            get_hyperdissipation_kx(kx, adaptive_rate)
        );

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to ky hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKy_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT adaptive_rate;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        const FLUCS_FLOAT hyperdissipation = (
            get_hyperdissipation_ky(ky, adaptive_rate)
        );

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to kz hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKz_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT adaptive_rate;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

        const FLUCS_FLOAT hyperdissipation = (
            get_hyperdissipation_kz(kz, adaptive_rate)
        );

        return hyperdissipation * functor(index);
    }
};


// Functor for calculating the total (perpendicular + directional)
// hyperdissipation for a given mode
template<typename FunctorT>
struct Hyperdissipation_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT adaptive_rate;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_FLOAT hyperdissipation = (
            get_hyperdissipation(index, adaptive_rate)
        );
        
        return hyperdissipation * functor(index);
    }
};

// Functor for registering the hyperdissipation functors for each component
template<typename FunctorT>
struct HyperdissipationSelector_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        switch (hyperdissipation_type) {
            case HYPERDISSIPATION_KZ_INT:
                return HyperdissipationKz_Functor<FunctorT>{
                    functor, adaptive_rate
                }(index);
            case HYPERDISSIPATION_KX_INT:
                return HyperdissipationKx_Functor<FunctorT>{
                    functor, adaptive_rate
                }(index);
            case HYPERDISSIPATION_KY_INT:
                return HyperdissipationKy_Functor<FunctorT>{
                    functor, adaptive_rate
                }(index);
            case HYPERDISSIPATION_KPERP_INT:
                return HyperdissipationKperp_Functor<FunctorT>{
                    functor, adaptive_rate
                }(index);
            default:
                __trap();
        }
    }
};
