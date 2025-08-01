#include <cupy/complex.cuh>

// Deal with float types

#ifdef DOUBLE_PRECISION

    #define FLOAT_TYPE double
    #define COMPLEX_TYPE complex<double>
    #define FLOAT_ABS(x) fabs(x)

#else

    #define FLOAT_TYPE float
    #define COMPLEX_TYPE complex<float>
    #define FLOAT_ABS(x) fabsf(x)

#endif


// Box sizes and grid points

#define Lx 1.0
#define Ly 1.0
#define Lz 1.0

#define NX 1
#define NY 1
#define NZ 1

#define HALF_NX 1
#define HALF_NY 1
#define HALF_NZ 1

// Array sizes

#define HALFUNPADDEDSIZE 1
#define HALFPADDEDSIZE 1
#define FULLPADDEDSIZE 1

// Other useful constants

#define TWOPI_OVER_LX (FLOAT_TYPE)1.0
#define TWOPI_OVER_LY (FLOAT_TYPE)1.0
#define TWOPI_OVER_LZ (FLOAT_TYPE)1.0
