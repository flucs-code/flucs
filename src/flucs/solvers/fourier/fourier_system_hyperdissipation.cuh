// Calculates the perpendicular hyperdissipation for a given kx, ky mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_perp(
    const FLUCS_FLOAT kx,
    const FLUCS_FLOAT ky,
    const FLUCS_FLOAT dt
) {

#ifdef HYPERDISSIPATION_PERP

    const FLUCS_FLOAT kperp2 = kx * kx + ky * ky;

#ifdef HYPERDISSIPATION_PERP_NORMALISED
    const FLUCS_FLOAT kx_max = kx_from_ikx(HALF_NX - 1);
    const FLUCS_FLOAT ky_max = ky_from_iky(HALF_NY - 1);
    const FLUCS_FLOAT kperp2_norm = kperp2 / (kx_max * kx_max + ky_max * ky_max);
#else
    const FLUCS_FLOAT kperp2_norm = kperp2;
#endif // NORMALISED

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_PERP;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_PERP_POWER; i++)
        hyperdissipation *= kperp2_norm;

    #ifdef HYPERDISSIPATION_PERP_ADAPTIVE
        hyperdissipation /= dt;
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
    const FLUCS_FLOAT dt
) {

#ifdef HYPERDISSIPATION_KX

    const FLUCS_FLOAT kx2 = kx * kx;

#ifdef HYPERDISSIPATION_KX_NORMALISED
    const FLUCS_FLOAT kx_max = kx_from_ikx(HALF_NX - 1);
    const FLUCS_FLOAT kx2_norm = kx2 / (kx_max * kx_max);
#else
    const FLUCS_FLOAT kx2_norm = kx2;
#endif

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KX;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KX_POWER; i++)
        hyperdissipation *= kx2_norm;

    #ifdef HYPERDISSIPATION_KX_ADAPTIVE
        hyperdissipation /= dt;
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
    const FLUCS_FLOAT dt
) {

#ifdef HYPERDISSIPATION_KY

    const FLUCS_FLOAT ky2 = ky * ky;

#ifdef HYPERDISSIPATION_KY_NORMALISED
    const FLUCS_FLOAT ky_max = ky_from_iky(HALF_NY - 1);
    const FLUCS_FLOAT ky2_norm = ky2 / (ky_max * ky_max);
#else
    const FLUCS_FLOAT ky2_norm = ky2;
#endif

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KY;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KY_POWER; i++)
        hyperdissipation *= ky2_norm;

    #ifdef HYPERDISSIPATION_KY_ADAPTIVE
        hyperdissipation /= dt;
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
    const FLUCS_FLOAT dt
) {

#ifdef HYPERDISSIPATION_KZ

    const FLUCS_FLOAT kz2 = kz * kz;

#ifdef HYPERDISSIPATION_KZ_NORMALISED
    const FLUCS_FLOAT kz_max = kz_from_ikz(HALF_NZ - 1);
    const FLUCS_FLOAT kz2_norm = kz2 / (kz_max * kz_max);
#else
    const FLUCS_FLOAT kz2_norm = kz2;
#endif

    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KZ;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KZ_POWER; i++)
        hyperdissipation *= kz2_norm;

    #ifdef HYPERDISSIPATION_KZ_ADAPTIVE
        hyperdissipation /= dt;
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
    const FLUCS_FLOAT dt
) {

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
    const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
    const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

    return get_hyperdissipation_perp(kx, ky, dt)
        + get_hyperdissipation_kx(kx, dt)
        + get_hyperdissipation_ky(ky, dt)
        + get_hyperdissipation_kz(kz, dt);
}

// Functor for calculating the size of the term due to perpendicular hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationPerp_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT dt;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_perp(kx, ky, dt);

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to kx hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKx_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT dt;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_kx(kx, dt);

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to ky hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKy_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT dt;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_ky(ky, dt);

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to kz hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKz_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT dt;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_kz(kz, dt);

        return hyperdissipation * functor(index);
    }
};


// Functor for calculating the total (perpendicular + directional)
// hyperdissipation for a given mode
template<typename FunctorT>
struct Hyperdissipation_Functor {
    const FunctorT functor;
    const FLUCS_FLOAT dt;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation(index, dt);
        return hyperdissipation * functor(index);
    }
};
