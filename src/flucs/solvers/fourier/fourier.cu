#include <cupy/complex.cuh>

// Deal with float types

#ifdef DOUBLE_PRECISION
    #define FLUCS_FLOAT double
    #define flucs_fabs(x) fabs(x)
#else
    #define FLUCS_FLOAT float
    #define flucs_fabs(x) fabsf(x)
#endif

#define FLUCS_COMPLEX complex<FLUCS_FLOAT>

// Box sizes and grid points
// #define TWOPI_OVER_LX (FLOAT_TYPE)1.0
// #define TWOPI_OVER_LY (FLOAT_TYPE)1.0
// #define TWOPI_OVER_LZ (FLOAT_TYPE)1.0
//
// // #define LX (FLOAT_TYPE)1.0
// // #define LY (FLOAT_TYPE)1.0
// // #define LZ (FLOAT_TYPE)1.0
//
// #define NX 1
// #define NY 1
// #define NZ 1
//
// #define HALF_NX 1
// #define HALF_NY 1
// #define HALF_NZ 1
//
// // Array sizes
//
// #define HALFUNPADDEDSIZE 1
// #define HALFPADDEDSIZE 1
// #define FULLPADDEDSIZE 1
