#include "div.h"
#include "pow.h"
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cudnn.h>
#include <iostream>

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

void test_div()
{
    auto run = [](float x, float y) {
        auto a = cuda_fdiv_rn(x, y);
        auto b = cuda_fdividef(x, y);
        std::cout << "cuda_fdiv_rn(" << x << ", " << y << ") == " << cuda_fdiv_rn(x, y) << std::endl;
        std::cout << "cuda_fdividef(" << x << ", " << y << ") == " << cuda_fdividef(x, y) << std::endl;
        std::cout << "" << x << " / " << y << " == " << x / y << std::endl;
        std::cout << std::endl;
    };
    run(FLT_MAX, FLT_MAX);
    run(3, 2);
}

int main(void)
{
    if (!init_cuda())
        return -1;

    test_pow();
    test_div();
    return 0;
}
