#include "SHMatrix.h"

int my_ceilf_division(float a, float b) {
  return 1 + ((a - 1) / b);
}

__global__ void FloatCUDAMemset_GPUKernel(float *d_array,
                                          int array_size, float val) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  d_array[idx % array_size] = val;
}

__global__ void ScaleUniformSHMatrix_GPUKernel(float *d_array, 
                                               int array_size,
                                               float lower, 
                                               float abs_range) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  d_array[idx] = lower + (d_array[idx] * abs_range);
}

void FloatCUDAMemset(float *d_array, int array_size, float val) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  FloatCUDAMemset_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_array, array_size, val);
}

void ScaleUniformSHMatrix(float *d_array, int array_size,
                          float lower, float higher) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ScaleUniformSHMatrix_GPUKernel << <num_threadblocks,
    threadblock_size >> >
    (d_array, array_size, lower, higher - lower);
}