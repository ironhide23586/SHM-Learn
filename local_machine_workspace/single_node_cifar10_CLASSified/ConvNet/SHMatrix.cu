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

__global__ void ElemwiseMultiplyInPlaceGPU_GPUKernel(float *d_a,
                                                     float *d_b,
                                                     int array_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  d_a[idx] *= d_b[idx];
}

__global__ void ElemwiseAddInPlaceGPU_GPUKernel(float *d_a,
                                                float *d_b,
                                                int array_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  d_a[idx] += d_b[idx];
}

__global__ void ElemwiseSubtractInPlaceGPU_GPUKernel(float *d_a,
                                                     float *d_b,
                                                     int array_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  d_a[idx] -= d_b[idx];
}

__global__ void ElemwiseDivideInPlaceGPU_GPUKernel(float *d_a,
                                                   float *d_b,
                                                   int array_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  d_a[idx] /= d_b[idx];
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
  ScaleUniformSHMatrix_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_array, array_size, lower, higher - lower);
}

void ElemwiseMultiplyInPlaceGPU(float *d_src, float *d_arg,
                                int array_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseMultiplyInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, array_size);
}

void ElemwiseAddInPlaceGPU(float *d_src, float *d_arg,
                           int array_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseAddInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, array_size);
}

void ElemwiseSubtractInPlaceGPU(float *d_src, float *d_arg,
                                int array_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseSubtractInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, array_size);
}

void ElemwiseDivideInPlaceGPU(float *d_src, float *d_arg,
                              int array_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseDivideInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, array_size);
}


void ElemwiseMultiplyInPlaceCPU(float *d_src, float *d_arg,
                                int array_size) {
  for (int i = 0; i < array_size; i++) {
    d_src[i] *= d_arg[i];
  }
}

void ElemwiseAddInPlaceCPU(float *d_src, float *d_arg,
                           int array_size) {
  for (int i = 0; i < array_size; i++) {
    d_src[i] += d_arg[i];
  }
}

void ElemwiseSubtractInPlaceCPU(float *d_src, float *d_arg,
                                int array_size) {
  for (int i = 0; i < array_size; i++) {
    d_src[i] -= d_arg[i];
  }
}
void ElemwiseDivideInPlaceCPU(float *d_src, float *d_arg,
                              int array_size) {
  for (int i = 0; i < array_size; i++) {
    d_src[i] /= d_arg[i];
  }
}