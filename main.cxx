#include "div.h"
#include "pow.h"
#include <cuda_runtime.h>
// system header
#include <cassert>
#include <cfloat>
#include <cmath>
#include <iostream>

using namespace std;

bool init_cuda()
{
    int cuda_dev_count;
    cudaGetDeviceCount(&cuda_dev_count);
    cout << "cuda device count: " << cuda_dev_count << endl;
    if (cuda_dev_count == 0)
        return false;

    int dev = 0;
    cudaDeviceProp prop;
    assert(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    cout << "use GPU device " << dev << ": " << prop.name << endl;
    cout << "SM number: " << prop.multiProcessorCount << endl;
    cout << "Share memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << endl;
    cout << "Thread per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Thread per EM: " << prop.maxThreadsPerMultiProcessor << endl;
    cudaSetDevice(dev);
    cout << "cuda initialized." << endl;
    return true;
}

void test_pow()
{
    auto run = [](float x, float y) {
        cout << "cuda_pow(" << x << ", " << y << ") == " << cuda_pow(x, y) << endl;
        cout << "pow(" << x << ", " << y << ") == " << pow(x, y) << endl;
        cout << endl;
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
        cout << "cuda_fdiv_rn(" << x << ", " << y << ") == " << cuda_fdiv_rn(x, y) << endl;
        cout << "cuda_fdividef(" << x << ", " << y << ") == " << cuda_fdividef(x, y) << endl;
        cout << "" << x << " / " << y << " == " << x / y << endl;
        cout << endl;
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
