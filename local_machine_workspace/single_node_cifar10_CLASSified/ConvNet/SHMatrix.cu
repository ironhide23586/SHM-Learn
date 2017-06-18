#include "SHMatrix.h"

#define MULT_OP 0
#define DIV_OP 1

// transpose_op_sel (elemwise ops) -
// 0 -> no transpose
// 1 -> transpose on a
// 2 -> transpose on b

int my_ceilf_division(float a, float b) {
  return 1 + ((a - 1) / b);
}

__host__ __device__ int get_tranposed_2DLin_idx(int src_idx, int ld_src,
                                                int array_size) {
  int cols_src = array_size / ld_src;
  int j = src_idx / cols_src;
  int i = src_idx - (j * cols_src);
  return j + i * ld_src;
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
                                                     int lda, int ldb,
                                                     int array_size,
                                                     int transpose_op_sel) {
  int org_idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  int a_idx = org_idx, b_idx = org_idx;
  if (transpose_op_sel == 1) {
    a_idx = get_tranposed_2DLin_idx(org_idx, lda, array_size);
  }
  else if (transpose_op_sel == 2) {
    b_idx = get_tranposed_2DLin_idx(org_idx, ldb, array_size);
  }
  float tmp = d_a[a_idx] * d_b[b_idx];
  __syncthreads();
  d_a[org_idx] = tmp;
  if (transpose_op_sel == 1)
    d_a[a_idx] = tmp;
}

__global__ void ElemwiseDivideInPlaceGPU_GPUKernel(float *d_a,
                                                   float *d_b,
                                                   int lda, int ldb,
                                                   int array_size,
                                                   int transpose_op_sel) {
  int org_idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  int a_idx = org_idx, b_idx = org_idx;
  if (transpose_op_sel == 1) {
    a_idx = get_tranposed_2DLin_idx(org_idx, lda, array_size);
  }
  else if (transpose_op_sel == 2) {
    b_idx = get_tranposed_2DLin_idx(org_idx, ldb, array_size);
  }
  float tmp = d_a[a_idx] / d_b[b_idx];
  __syncthreads();
  d_a[org_idx] = tmp;
  if (transpose_op_sel == 1)
    d_a[a_idx] = tmp;
}

__global__ void ElemwiseAddInPlaceGPU_GPUKernel(float *d_a,
                                                float *d_b,
                                                int lda, int ldb,
                                                int array_size,
                                                int transpose_op_sel) {
  int org_idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  int a_idx = org_idx, b_idx = org_idx;
  if (transpose_op_sel == 1) {
    a_idx = get_tranposed_2DLin_idx(org_idx, lda, array_size);
  }
  else if (transpose_op_sel == 2) {
    b_idx = get_tranposed_2DLin_idx(org_idx, ldb, array_size);
  }
  float tmp = d_a[a_idx] + d_b[b_idx];
  __syncthreads();
  d_a[org_idx] = tmp;
  if (transpose_op_sel == 1)
    d_a[a_idx] = tmp;
}

__global__ void ElemwiseSubtractInPlaceGPU_GPUKernel(float *d_a,
                                                     float *d_b,
                                                     int lda, int ldb,
                                                     int array_size,
                                                     int transpose_op_sel) {
  int org_idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  int a_idx = org_idx, b_idx = org_idx;
  if (transpose_op_sel == 1) {
    a_idx = get_tranposed_2DLin_idx(org_idx, lda, array_size);
  }
  else if (transpose_op_sel == 2) {
    b_idx = get_tranposed_2DLin_idx(org_idx, ldb, array_size);
  }
  float tmp = d_a[a_idx] - d_b[b_idx];
  __syncthreads();
  d_a[org_idx] = tmp;
  if (transpose_op_sel == 1)
    d_a[a_idx] = tmp;
}

__global__  void ElemwiseAddInPlaceGPU_Scalar_GPUKernel(float *d_a,
                                                        float scalar,
                                                        int array_size) {
  int org_idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  float tmp = d_a[org_idx] + scalar;
  __syncthreads();
  d_a[org_idx] = tmp;
}

__global__  void ElemwiseSubtractInPlaceGPU_Scalar_GPUKernel(float *d_a,
                                                             float scalar,
                                                             int array_size) {
  int org_idx = (blockDim.x * blockIdx.x + threadIdx.x) % array_size;
  float tmp = d_a[org_idx] - scalar;
  __syncthreads();
  d_a[org_idx] = tmp;
}

int get_transpose_op_sel(bool src_T_op, bool arg_T_op) {
  int transpose_op_sel = 0;
  if (src_T_op != arg_T_op)
    transpose_op_sel = src_T_op ? 1 : 2;
  return transpose_op_sel;
}

void op_worker(float *d_src, float *d_arg,
               int ld_src, int ld_arg,
               int array_size, int transpose_op_sel,
               ELEM_OP elem_op) {
  float *tmp = (float *)malloc(sizeof(float) * array_size);
  int src_idx, arg_idx;
  for (int org_idx = 0; org_idx < array_size; org_idx++) {
    src_idx = org_idx;
    arg_idx = org_idx;
    if (transpose_op_sel == 1) {
      src_idx = get_tranposed_2DLin_idx(src_idx, ld_src, array_size);
    }
    else if (transpose_op_sel == 2) {
      arg_idx = get_tranposed_2DLin_idx(arg_idx, ld_arg, array_size);
    }
    if (elem_op == MULT)
      tmp[org_idx] = d_src[src_idx] * d_arg[arg_idx];
    else if (elem_op == DIV)
      tmp[org_idx] = d_src[src_idx] / d_arg[arg_idx];
    else if (elem_op == ADD)
      tmp[org_idx] = d_src[src_idx] + d_arg[arg_idx];
    else if (elem_op == SUB)
      tmp[org_idx] = d_src[src_idx] - d_arg[arg_idx];
  }
  for (int org_idx = 0; org_idx < array_size; org_idx++) {
    if (transpose_op_sel == 1) {
      src_idx = org_idx;
      src_idx = get_tranposed_2DLin_idx(src_idx, ld_src, array_size);
      d_src[src_idx] = tmp[org_idx];
    }
    else {
      d_src[org_idx] = tmp[org_idx];
    }
  }
  free(tmp);
}

void FloatCUDAMemset(float *d_array, int array_size, float val) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  FloatCUDAMemset_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_array, array_size, val);
  CudaCheckError();
}

void ScaleUniformSHMatrix(float *d_array, int array_size,
                          float lower, float higher) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ScaleUniformSHMatrix_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_array, array_size, lower, higher - lower);
  CudaCheckError();
}

void ElemwiseMultiplyInPlaceGPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_T_op,
                                bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseMultiplyInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, ld_src, ld_arg,
                           array_size, transpose_op_sel);
  CudaCheckError();
}

void ElemwiseDivideInPlaceGPU(float *d_src, float *d_arg,
                              int ld_src, int ld_arg,
                              int array_size, bool src_T_op,
                              bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseDivideInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, ld_src, ld_arg,
                           array_size, transpose_op_sel);
  CudaCheckError();
}

void ElemwiseAddInPlaceGPU(float *d_src, float *d_arg,
                           int ld_src, int ld_arg,
                           int array_size, bool src_T_op,
                           bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseAddInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, ld_src, ld_arg,
                           array_size, transpose_op_sel);
  CudaCheckError();
}

void ElemwiseSubtractInPlaceGPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_T_op,
                                bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseSubtractInPlaceGPU_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, d_arg, ld_src, ld_arg,
                           array_size, transpose_op_sel);
  CudaCheckError();
}

void ElemwiseAddInPlaceGPU_Scalar(float *d_src, float scalar,
                                  int array_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseAddInPlaceGPU_Scalar_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, scalar, array_size);
  CudaCheckError();
}

void ElemwiseSubtractInPlaceGPU_Scalar(float *d_src, float scalar,
                                       int array_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division(array_size, threadblock_size);
  ElemwiseSubtractInPlaceGPU_Scalar_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_src, scalar, array_size);
  CudaCheckError();
}

void ElemwiseMultiplyInPlaceCPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_T_op,
                                bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  op_worker(d_src, d_arg, ld_src, ld_arg, array_size, transpose_op_sel, MULT);
}

void ElemwiseDivideInPlaceCPU(float *d_src, float *d_arg,
                              int ld_src, int ld_arg,
                              int array_size, bool src_T_op,
                              bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  op_worker(d_src, d_arg, ld_src, ld_arg, array_size, transpose_op_sel, DIV);
}

void ElemwiseAddInPlaceCPU(float *d_src, float *d_arg,
                           int ld_src, int ld_arg,
                           int array_size, bool src_T_op,
                           bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  op_worker(d_src, d_arg, ld_src, ld_arg, array_size, transpose_op_sel, ADD);
}

void ElemwiseSubtractInPlaceCPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_T_op,
                                bool arg_T_op) {
  int transpose_op_sel = get_transpose_op_sel(src_T_op, arg_T_op);
  op_worker(d_src, d_arg, ld_src, ld_arg, array_size, transpose_op_sel, SUB);
}