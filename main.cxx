#include <cassert>
#include <cudnn.h>
#include <iostream>
#include <cmath>
#include "pow.h"

bool init_cuda()
{
    int cuda_dev_count;
    cudaGetDeviceCount(&cuda_dev_count);
    std::cout << "cuda device count: " << cuda_dev_count << std::endl;
    if (cuda_dev_count == 0)
        return false;

    int dev = 0;
    cudaDeviceProp prop;
    assert(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    std::cout << "use GPU device " << dev << ": " << prop.name << std::endl;
    std::cout << "SM number: " << prop.multiProcessorCount << std::endl;
    std::cout << "Share memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Thread per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Thread per EM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    cudaSetDevice(dev);
    std::cout << "cuda initialized." << std::endl;
    return true;
}

void test_pow()
{
    auto run = [](float x, float y) {
        std::cout << "cuda_pow(" << x << ", " << y << ") == " << cuda_pow(x, y) << std::endl;
        std::cout << "std::pow(" << x << ", " << y << ") == " << std::pow(x, y) << std::endl;
        std::cout << std::endl;
    };
    run(2, 4);
    run(-2, 3);
    run(2, 0.3);
    run(-2, 0.3);
}

int main(void)
{
    if (!init_cuda())
        return -1;

    test_pow();
    return 0;
}
