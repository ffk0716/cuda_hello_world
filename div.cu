#include "cudnn.h"

__global__ void fdiv_rn_global(float x, float y, float* r)
{
    *r = __fdiv_rn(x, y);
}

float cuda_fdiv_rn(float x, float y)
{

    float *gpu_result, result;
    cudaMalloc((void**)&gpu_result, sizeof(float));
    fdiv_rn_global<<<1, 1>>>(x, y, gpu_result);
    cudaMemcpy(&result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_result);
    // cudaDeviceReset(); // force printf flush
    return result;
}

__global__ void fdividef_global(float x, float y, float* r)
{
    *r = __fdividef(x, y);
}

float cuda_fdividef(float x, float y)
{

    float *gpu_result, result;
    cudaMalloc((void**)&gpu_result, sizeof(float));
    fdividef_global<<<1, 1>>>(x, y, gpu_result);
    cudaMemcpy(&result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_result);
    // cudaDeviceReset(); // force printf flush
    return result;
}
