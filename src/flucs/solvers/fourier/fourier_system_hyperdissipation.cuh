// Calculates the perpendicular hyperdissipation for a given kx, ky mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_perp(const FLUCS_FLOAT kx, const FLUCS_FLOAT ky) {

#ifdef HYPERDISSIPATION_PERP

    const FLUCS_FLOAT kperp2 = kx * kx + ky * ky;
    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_PERP;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_PERP_POWER; i++)
        hyperdissipation *= kperp2;

    return hyperdissipation;

#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the kx hyperdissipation for a given kx mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_kx(const FLUCS_FLOAT kx) {

#ifdef HYPERDISSIPATION_KX

    const FLUCS_FLOAT kx2 = kx * kx;
    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KX;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KX_POWER; i++)
        hyperdissipation *= kx2;

    return hyperdissipation;
#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the ky hyperdissipation for a given ky mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_ky(const FLUCS_FLOAT ky) {

#ifdef HYPERDISSIPATION_KY

    const FLUCS_FLOAT ky2 = ky * ky;
    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KY;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KY_POWER; i++)
        hyperdissipation *= ky2;

    return hyperdissipation;
#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the kz hyperdissipation for a given kz mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation_kz(const FLUCS_FLOAT kz) {

#ifdef HYPERDISSIPATION_KZ

    const FLUCS_FLOAT kz2 = kz * kz;
    FLUCS_FLOAT hyperdissipation = HYPERDISSIPATION_KZ;

    #pragma unroll
    for (int i = 0; i < HYPERDISSIPATION_KZ_POWER; i++)
        hyperdissipation *= kz2;

    return hyperdissipation;
#else
    return (FLUCS_FLOAT)0;
#endif
}

// Calculates the total hyperdissipation for a given mode
__device__ __forceinline__
FLUCS_FLOAT get_hyperdissipation(const size_t index) {

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
    const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
    const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

    return get_hyperdissipation_perp(kx, ky)
        + get_hyperdissipation_kx(kx)
        + get_hyperdissipation_ky(ky)
        + get_hyperdissipation_kz(kz);
}

// Functor for calculating the size of the term due to perpendicular hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationPerp_Functor {
    const FunctorT functor;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_perp(kx, ky);

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to kx hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKx_Functor {
    const FunctorT functor;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_kx(kx);

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to ky hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKy_Functor {
    const FunctorT functor;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_ky(ky);

        return hyperdissipation * functor(index);
    }
};

// Functor for calculating the size of the term due to kz hyperdissipation for a given mode
template<typename FunctorT>
struct HyperdissipationKz_Functor {
    const FunctorT functor;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation_kz(kz);

        return hyperdissipation * functor(index);
    }
};


// Functor for calculating the total (perpendicular + directional)
// hyperdissipation for a given mode
template<typename FunctorT>
struct Hyperdissipation_Functor {
    const FunctorT functor;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_FLOAT hyperdissipation = get_hyperdissipation(index);
        return hyperdissipation * functor(index);
    }
};
