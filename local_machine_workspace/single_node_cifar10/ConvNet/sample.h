#include <cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>

//#include "sample.cu"

__global__ void addKernel(int *c, const int *a, const int *b);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void func(float *d_pnt);

void foo(float *d_pnt);