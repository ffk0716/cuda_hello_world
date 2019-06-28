#include <stdio.h>
#include "cudnn.h"

__global__ void cuda_pow_g(float x, float y) {
    auto r = __powf(x, y);
    printf("pow(%f, %f) == %f\n", x, y, r);
}

void cuda_pow(float x, float y)
{
    cuda_pow_g<<<1, 1>>>(x, y);
    cudaDeviceReset();
}
