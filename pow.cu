#include "cudnn.h"

__global__ void powf_global(float x, float y, float* r)
{
    *r = __powf(x, y);
}

float cuda_pow(float x, float y)
{

    float *gpu_result, result;
    cudaMalloc((void**)&gpu_result, sizeof(float));
    powf_global<<<1, 1>>>(x, y, gpu_result);
    cudaMemcpy(&result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_result);
    // cudaDeviceReset(); // force printf flush
    return result;
}
