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

// transpose_op_sel -
// 0 -> no transpose
// 1 -> transpose on a
// 2 -> transpose on b
__global__ void ElemwiseMultiplyInPlaceGPU_GPUKernel(float *d_a,
                                                     float *d_b,
                                                     int lda, int ldb,
                                                     int array_size,
                                                     int transpose_op_sel) {
  int org_idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  int a_idx = org_idx;
  int b_idx = org_idx;
  if (transpose_op_sel == 1) {
    int cols_a = array_size / lda;
    int j = a_idx / cols_a;
    int i = a_idx - (j * cols_a);
    a_idx = j + i * lda;
  }
  else if (transpose_op_sel == 2) {
    int cols_b = array_size / ldb;
    int j = b_idx / cols_b;
    int i = b_idx - (j * cols_b);
    b_idx = j + i * ldb;
  }
  float tmp = d_a[a_idx] * d_b[b_idx];
  __syncthreads();
  d_a[org_idx] = tmp;
  if (transpose_op_sel == 1)
    d_a[a_idx] = tmp;
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
                                int ld_src, int ld_arg,
                                int array_size, bool src_T_op,
                                bool arg_T_op) {
  int transpose_op_sel;
  if (src_T_op == arg_T_op)
    transpose_op_sel = 0;
  else
    transpose_op_sel = src_T_op ? 1 : 2;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseMultiplyInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, ld_src, ld_arg,
                           array_size, transpose_op_sel);
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
                                int lda, int ldb,
                                int array_size, bool src_T_op,
                                bool arg_T_op) {
  int transpose_op_sel;
  if (src_T_op == arg_T_op)
    transpose_op_sel = 0;
  else
    transpose_op_sel = src_T_op ? 1 : 2;
  float *tmp = (float *)malloc(sizeof(float) * array_size);
  for (int org_idx = 0; org_idx < array_size; org_idx++) {
    int a_idx = org_idx;
    int b_idx = org_idx;
    if (transpose_op_sel == 1) {
      int cols_a = array_size / lda;
      int j = a_idx / cols_a;
      int i = a_idx - (j * cols_a);
      a_idx = j + i * lda;
    }
    else if (transpose_op_sel == 2) {
      int cols_b = array_size / ldb;
      int j = b_idx / cols_b;
      int i = b_idx - (j * cols_b);
      b_idx = j + i * ldb;
    }
    tmp[org_idx] = d_src[a_idx] * d_arg[b_idx];
  }
  for (int org_idx = 0; org_idx < array_size; org_idx++) {
    if (transpose_op_sel == 1) {
      int a_idx = org_idx;
      int cols_a = array_size / lda;
      int j = a_idx / cols_a;
      int i = a_idx - (j * cols_a);
      a_idx = j + i * lda;
      d_src[a_idx] = tmp[org_idx];
    }
    else {
      d_src[org_idx] = tmp[org_idx];
    }
  }
  free(tmp);
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