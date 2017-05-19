#include "SHMatrix.h"

int my_ceilf_division(float a, float b) {
  return 1 + ((a - 1) / b);
}

__global__ void FloatCUDAMemset_GPUKernel(float *d_array,
                                         int array_size, float val) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  d_array[idx % array_size] = val;
}

void FloatCUDAMemset(float *d_array, int array_size, float val) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  FloatCUDAMemset_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_array, array_size, val);
}