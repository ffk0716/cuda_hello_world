#include <pow.h>

int main(void)
{
    cuda_pow(2, 3);
    cuda_pow(-2, 3);
    cuda_pow(2, 0.3);
    cuda_pow(-2, 0.3);
    return 0;
}
