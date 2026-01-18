#pragma once

__device__ float atomicMaxFloat(float* addr, float value) {
    int* address_as_int = (int*) addr;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ double atomicMaxFloat(double* addr, double value)
{
    unsigned long long* address_as_ull =
        reinterpret_cast<unsigned long long*>(addr);

    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;

        double assumed_val = __longlong_as_double(assumed);
        double max_val = fmax(value, assumed_val);

        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(max_val)
        );

    } while (assumed != old);

    return __longlong_as_double(old);
}


