#include "ConvLayer.h"
#include "math_functions.h"

int my_ceilf_division_ConvLayer(float a, float b) {
  return 1 + ((a - 1) / b);
}

__global__ void WeightMatrixRegularizeElemWiseConv_GPUKernel(float *d_mat_in,
                                                             float reg_inp_scalar,
                                                             int d_mat_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < d_mat_size)
    d_mat_in[idx] -= reg_inp_scalar * d_mat_in[idx] * d_mat_in[idx];
}

__global__ void SubtractElemwise_Conv_GPUKernel(float *d_mat, float delta,
                                           int total_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x);
  if (idx < total_size)
    d_mat[idx] -= delta;
}

void SubtractElemwise_Conv(float *d_mat, float delta, int mat_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_ConvLayer(mat_size, threadblock_size);
  SubtractElemwise_Conv_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_mat, delta, mat_size);
}

void WeightMatrixRegularizeElemWiseConv(float *d_mat_in,
                                        float reg_inp_scalar, int d_mat_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_ConvLayer(d_mat_size, threadblock_size);
  WeightMatrixRegularizeElemWiseConv_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_mat_in, reg_inp_scalar,
     d_mat_size);
}