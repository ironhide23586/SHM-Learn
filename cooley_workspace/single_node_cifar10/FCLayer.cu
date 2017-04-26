#include "FCLayer.h"
#include "math_functions.h"

void print_d_var2(float *d_v, int r, int c, bool print_elem = true) {
  std::cout << "*****************************" << std::endl;
  float *h_v = (float *)malloc(sizeof(float) * r * c);
  cudaMemcpy(h_v, d_v, sizeof(float) * r * c, cudaMemcpyDeviceToHost);
  float mini = h_v[0], maxi = h_v[0];
  int mini_idx = 0, maxi_idx = 0;
  float sum = 0.0;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if (print_elem)
        printf("%f\t", h_v[j + i * c]);
      if (h_v[j + i * c] < mini) {
        mini = h_v[j + i * c];
        mini_idx = j + i * c;
      }
      if (h_v[j + i * c] > maxi) {
        maxi = h_v[j + i * c];
        maxi_idx = j + i * c;
      }
      sum += h_v[j + i * c];
    }
    if (print_elem)
      std::cout << std::endl;
  }
  std::cout << "Shape = (" << r << ", " << c << ")" << std::endl;
  std::cout << "Minimum at index " << mini_idx << " = " << mini << std::endl;
  std::cout << "Maximum at index " << maxi_idx << " = " << maxi << std::endl;
  std::cout << "Average of all elements = " << sum / (r * c) << std::endl;
  free(h_v);
}

int my_ceilf_division_FCLayer(float a, float b) {
  return 1 + ((a - 1) / b);
}

__global__ void FloatGPUMemset_GPUKernel(float *d_array,
                                         int array_size, float val) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  d_array[idx % array_size] = val;
}

//__global__ void ReAlignMemory_ShiftRight_GPUKernel(float *d_mat, 
//                                                   int total_size,
//                                                   int elem_size,
//                                                   int num_elems) {
//  int compact_size = total_size - num_elems;
//  int read_idx = (threadIdx.x % compact_size);
//  int write_idx = (read_idx + ceil((float) read_idx / elem_size));
//  if (read_idx % elem_size == 0)
//    write_idx++;
//  float val_to_copy = d_mat[read_idx];
//  __syncthreads();
//  d_mat[write_idx] = val_to_copy;
//}

//__global__ void ReAlignMemory_ShiftLeft_GPUKernel(float *d_data,
//                                                  int total_size,
//                                                  int rows,
//                                                  int cols) {
//  int compact_size = total_size - rows;
//  int write_idx = (threadIdx.x % compact_size);
//  int targ_cols = cols - 1;
//  int read_idx = write_idx + ceil((float)write_idx / targ_cols);
//  if (write_idx % targ_cols == 0)
//    read_idx++;
//  float val_to_copy = d_data[read_idx];
//  __syncthreads();
//  d_data[write_idx] = val_to_copy;
//}

__global__ void FillOnes_GPUKernel(float *d_data, int elem_size,
                                   int num_batches) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % (num_batches + 1);
  d_data[idx * elem_size] = 1.0f;
}

__global__ void InitIdentityMatrix_GPUKernel(float *d_mat, int d_mat_side) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % (d_mat_side * d_mat_side);
  d_mat[idx] = (idx % (d_mat_side + 1)) == 0 ? 1.0f : 0.0f;
}

__global__ void WeightMatrixRegularizeElemWise_GPUKernel(float *d_mat_in,
                                                         int d_mat_cols,
                                                         float reg_inp_scalar,
                                                         int d_mat_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < d_mat_size && idx >= d_mat_cols)
    //d_mat_in[idx] -= (reg_inp_scalar * d_mat_in[idx] * d_mat_in[idx]);
    d_mat_in[idx] *= reg_inp_scalar;
}

__global__ void ElemwiseGradCompute_GPUKernel(float *d_data,
                                              float *d_out_minus_labels,
                                              float *d_elem_grads,
                                              int input_batch_size,
                                              int input_neurons,
                                              int output_neurons) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x)
    % (input_batch_size * (input_neurons + 1) * output_neurons);
  int idx_y = idx / output_neurons;
  int d_data_idx_T_y = idx_y / input_batch_size;
  int d_data_idx_T_x = idx_y - (d_data_idx_T_y * input_batch_size);
  d_elem_grads[idx] = d_data[d_data_idx_T_y
    + d_data_idx_T_x * (input_neurons + 1)]
    * d_out_minus_labels[idx - d_data_idx_T_y
    * input_batch_size
    * output_neurons];
}

__global__ void
ComputeGradientsFromElemGrads_GPUKernel(float *d_elem_grads,
                                        float *d_softmax_gradients,
                                        float learning_rate,
                                        float momentum,
                                        int input_batch_size,
                                        int input_neurons,
                                        int output_neurons) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x)
    % ((input_neurons + 1) * output_neurons);
  int idx_y = idx / output_neurons;
  int idx_x = idx - idx_y * output_neurons;
  int idx_elem_grads_y = idx_y * input_batch_size;
  float sum = 0.0f;
  for (int i = 0; i < input_batch_size; i++) {
    sum += d_elem_grads[idx_x + (idx_elem_grads_y + i) * output_neurons];
  }
  d_softmax_gradients[idx] = (learning_rate / input_batch_size) * sum
    - momentum * d_softmax_gradients[idx];
}

__global__ void ComputeSoftmaxLoss_GPUKernel(float *d_out, float *d_labels,
                                             float *d_out_minus_labels,
                                             float coeff,
                                             int input_batch_size,
                                             int output_neurons) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x)
    % (input_batch_size * output_neurons);
  //d_out_minus_labels[idx] = coeff * (d_out[idx] - d_labels[idx]);
  if (d_labels[idx] == 1)
    d_out_minus_labels[idx] = coeff * (1.0f - d_out[idx]);
  else
    d_out_minus_labels[idx] = coeff * d_out[idx];
}

__global__ void ReluBackprop_GPUKernel(float *d_backprop_derivatives,
                                       float *d_out_xw_act,
                                       float *d_fwd_layer_derivatives,
                                       float relu_clip,
                                       int derivative_matrix_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % derivative_matrix_size;
  d_fwd_layer_derivatives[idx] = d_out_xw_act[idx] > relu_clip
    ? d_backprop_derivatives[idx] : relu_clip;
}

__global__ void SigmoidBackprop_GPUKernel(float *d_backprop_derivatives,
                                          float *d_out_xw_act,
                                          float *d_fwd_layer_derivatives,
                                          int derivative_matrix_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % derivative_matrix_size;
  d_fwd_layer_derivatives[idx] = d_out_xw_act[idx]
    * (1.0f - d_out_xw_act[idx])
    * d_backprop_derivatives[idx];
}

__global__ void ReplaceVal_GPUKernel(float *d_mat, int total_size,
                                     float val, float replace_val) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % total_size;
  if (d_mat[idx] == val)
    d_mat[idx] = replace_val;
}

__global__ void SubtractElemwise_GPUKernel(float *d_mat, float delta,
                                           int total_size) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x);
  if (idx < total_size)
    d_mat[idx] -= delta;
}

__global__ void Replace2Vals_GPUKernel(float *d_mat, int total_size,
                                       float val0, float val1,
                                       float replace_val0, float replace_val1) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % total_size;
  if (d_mat[idx] < val0)
    d_mat[idx] = replace_val0;
  else if (d_mat[idx] > val1)
    d_mat[idx] = replace_val1;
}

__global__ void ShiftRight_PopulateHelper_GPUKernel(float *d_mat,
                                                    float *d_helper,
                                                    int damaged_elems,
                                                    int rows, int cols) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % damaged_elems;
  int i = floor(0.5f * (sqrt((float)1 + 8 * idx) - 1.0f)) + 1;
  int j = idx - i * (i - 1) / 2;
  int read_idx = j + i * cols;
  d_helper[idx] = d_mat[read_idx];
}

__global__ void ReAlignMemory_ShiftRight_GPUKernel(float *d_mat,
                                                   float *d_helper,
                                                   int total_size,
                                                   int cols,
                                                   int thread_chunk_size) {
  extern __shared__ float read_vals[];
  int shared_mem_idx, read_idx, read_idx_row, write_idx;
  int row_linear_idx = blockIdx.x * cols;
  int read_idx_base = row_linear_idx
    + (threadIdx.x * thread_chunk_size) % cols;
  int row_last_linear_idx = row_linear_idx + cols;
  for (read_idx = read_idx_base; read_idx < row_last_linear_idx;
       read_idx++) {
    read_idx_row = read_idx / cols;
    shared_mem_idx = read_idx - row_linear_idx;
    if (read_idx >= read_idx_row * (1 + cols)) {
      read_vals[shared_mem_idx] = d_mat[read_idx];
    }
    else {
      read_vals[shared_mem_idx] = d_helper[read_idx - cols * read_idx_row
        + (read_idx_row - 1)
        * read_idx_row / 2];
    }
  }
  __syncthreads();
  for (read_idx = read_idx_base; read_idx < row_last_linear_idx;
       read_idx++) {
    write_idx = (read_idx + ceil((float)read_idx / cols)) + !(read_idx % cols);
    d_mat[write_idx] = read_vals[read_idx - row_linear_idx];
    if ((write_idx - 1) % (cols + 1) == 0) {
      d_mat[write_idx - 1] = 1.0f;
    }
  }
}

void ReAlignMemory_ShiftRight(float *d_mat, float *d_helper,
                              int rows, int cols, int max_threadblock_size) {
  int org_size = rows * cols;
  int reqd_threads = rows * (rows - 1) / 2;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  if (threadblock_size > max_threadblock_size)
    threadblock_size = max_threadblock_size;
  int num_threadblocks = my_ceilf_division_FCLayer(reqd_threads, threadblock_size);
  int thread_chunk_size = my_ceilf_division_FCLayer(cols, max_threadblock_size);
  ShiftRight_PopulateHelper_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_mat, d_helper, reqd_threads,
     rows, cols);
  reqd_threads = my_ceilf_division_FCLayer(cols, thread_chunk_size);
  threadblock_size = my_ceilf_division_FCLayer(reqd_threads, GPU_WARP_SIZE)
    * GPU_WARP_SIZE;
  ReAlignMemory_ShiftRight_GPUKernel << < rows,
    threadblock_size,
    sizeof(float) * cols >> >
    (d_mat, d_helper,
     org_size, cols,
     thread_chunk_size);
}

__global__ void ShiftLeft_PopulateHelper_GPUKernel(float *d_mat,
                                                   float *d_helper,
                                                   int damaged_elems,
                                                   int rows, int cols) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) % damaged_elems;
  int i = floor(0.5f * (sqrt((float)1 + 8 * idx) - 1.0f));
  int j = cols - (idx - i * (i - 1) / 2) - 1 + i;
  int read_idx = j + i * cols;
  d_helper[idx] = d_mat[read_idx];
}

__global__ void ReAlignMemory_ShiftLeft_GPUKernel(float *d_mat,
                                                  float *d_helper,
                                                  int total_size,
                                                  int cols,
                                                  int thread_chunk_size) {
  extern __shared__ float read_vals[];
  int shared_mem_idx, read_idx, read_idx_row, write_idx;
  int row_linear_idx = blockIdx.x * cols;
  int rows = total_size / cols;
  int read_idx_base = row_linear_idx
    + (threadIdx.x * thread_chunk_size) % cols + 1;
  int read_idx_lateral;
  int row_last_linear_idx = row_linear_idx + cols;
  for (read_idx = read_idx_base; read_idx < row_last_linear_idx;
       read_idx++) {
    read_idx_row = read_idx / cols;
    shared_mem_idx = read_idx - row_linear_idx - 1;
    if (read_idx < ((read_idx_row + 1) * (cols - 1))
        || blockIdx.x == (rows - 1)) {
      read_vals[shared_mem_idx] = d_mat[read_idx];
    }
    else {
      read_idx_lateral = row_linear_idx + cols - shared_mem_idx - 2;
      read_vals[shared_mem_idx] = d_helper[read_idx_lateral
        - cols * read_idx_row
        + (read_idx_row - 1)
        * read_idx_row / 2 + read_idx_row];
    }
  }
  __syncthreads();
  for (read_idx = read_idx_base; read_idx < row_last_linear_idx;
       read_idx++) {
    read_idx_row = read_idx / cols;
    shared_mem_idx = read_idx - row_linear_idx - 1;
    write_idx = row_linear_idx + shared_mem_idx - read_idx_row;
    d_mat[write_idx] = read_vals[shared_mem_idx];
  }
}

void ReAlignMemory_ShiftLeft(float *d_mat, float *d_helper,
                             int rows, int cols, int max_threadblock_size) {
  int org_size = rows * cols;
  int reqd_threads = rows * (rows - 1) / 2;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  if (threadblock_size > max_threadblock_size)
    threadblock_size = max_threadblock_size;
  int num_threadblocks = my_ceilf_division_FCLayer(reqd_threads, threadblock_size);
  int thread_chunk_size = my_ceilf_division_FCLayer((cols - 1), max_threadblock_size);
  ShiftLeft_PopulateHelper_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_mat, d_helper, reqd_threads,
     rows, cols);
  reqd_threads = my_ceilf_division_FCLayer((cols - 1), thread_chunk_size);
  threadblock_size = my_ceilf_division_FCLayer(reqd_threads, GPU_WARP_SIZE)
    * GPU_WARP_SIZE;
  ReAlignMemory_ShiftLeft_GPUKernel << < rows,
    threadblock_size,
    sizeof(float) * (cols - 1) >> >
    (d_mat, d_helper,
     org_size, cols,
     thread_chunk_size);
}

void ReAlignMemory_ShiftLeft_CPU(float *d_mat, int rows, int cols) {
  int sz = rows * (cols - 1);
  float *tmp0 = (float *)malloc(sizeof(float) * rows * cols);
  float *tmp1 = (float *)malloc(sizeof(float) * sz);
  cudaMemcpy(tmp0, d_mat, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  for (int i = 0; i < rows; i++) {
    for (int j = 1; j < cols; j++) {
      tmp1[(j - 1) + i * (cols - 1)] = tmp0[j + i * cols];
    }
  }
  cudaMemcpy(d_mat, tmp1, sizeof(float) * sz, cudaMemcpyHostToDevice);
  free(tmp0);
  free(tmp1);
}

void SubtractElemwise(float *d_mat, float delta, int mat_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(mat_size, threadblock_size);
  SubtractElemwise_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_mat, delta, mat_size);
}

void FloatGPUMemset(float *d_array, int array_size, float val) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(array_size, threadblock_size);
  FloatGPUMemset_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_array, array_size, val);
}

void ReplaceVal(float *d_mat, int total_size, float val, float replace_val) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(total_size, threadblock_size);
  ReplaceVal_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_mat, total_size, val, replace_val);
}

void Replace2Vals(float *d_mat, int total_size, float val0, float val1,
                  float replace_val0, float replace_val1) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(total_size, threadblock_size);
  Replace2Vals_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_mat, total_size, val0, val1,
     replace_val0, replace_val1);
}

void FillOnes(float *d_data, int batch_size, int elem_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(batch_size, threadblock_size);
  FillOnes_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_data, elem_size + 1,
                           batch_size);
}

void InitIdentityMatrix(float *d_mat, int side) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer((side * side), threadblock_size);
  InitIdentityMatrix_GPUKernel << < num_threadblocks,
    threadblock_size >> > (d_mat, side);
}

void WeightMatrixRegularizeElemWise(float *d_mat_in, int d_mat_cols,
                                    float reg_inp_scalar, int d_mat_size) {
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(d_mat_size, threadblock_size);
  WeightMatrixRegularizeElemWise_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_mat_in, d_mat_cols,
     reg_inp_scalar, d_mat_size);
}

void ElemwiseGradCompute(float *d_data, float *d_out_minus_labels,
                         float *d_elem_grads, int input_batch_size,
                         int input_neurons, int output_neurons) {
  int reqd_threads = (input_batch_size * (input_neurons + 1))
    * output_neurons;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(reqd_threads, threadblock_size);
  ElemwiseGradCompute_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_data, d_out_minus_labels, d_elem_grads,
     input_batch_size, input_neurons,
     output_neurons);
}

void ComputeGradientsFromElemGrads(float *d_elem_grads,
                                   float *d_softmax_gradients,
                                   float learning_rate, float momentum,
                                   int input_batch_size, int input_neurons,
                                   int output_neurons) {
  int reqd_threads = (input_neurons + 1) * output_neurons;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(reqd_threads, threadblock_size);

  ComputeGradientsFromElemGrads_GPUKernel << < num_threadblocks,
    threadblock_size >> >
    (d_elem_grads,
     d_softmax_gradients,
     learning_rate, momentum,
     input_batch_size, input_neurons,
     output_neurons);
}

void ComputeSoftmaxLoss(float *d_out, float *d_labels,
                        float *d_out_minus_labels, float coeff,
                        int input_batch_size, int output_neurons) {
  int reqd_threads = input_batch_size * output_neurons;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(reqd_threads, threadblock_size);
  ComputeSoftmaxLoss_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_out, d_labels,
     d_out_minus_labels, coeff,
     input_batch_size, output_neurons);
}

void ReluBackprop(float *d_backprop_derivatives, float *d_out_xw_act,
                  float *d_fwd_layer_derivatives, float relu_clip,
                  int derivative_matrix_size) {
  int reqd_threads = derivative_matrix_size;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(reqd_threads, threadblock_size);
  ReluBackprop_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_backprop_derivatives, d_out_xw_act,
     d_fwd_layer_derivatives, relu_clip,
     derivative_matrix_size);
}

void SigmoidBackprop(float *d_backprop_derivatives, float *d_out_xw_act,
                     float *d_fwd_layer_derivatives,
                     int derivative_matrix_size) {
  int reqd_threads = derivative_matrix_size;
  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
  int num_threadblocks = my_ceilf_division_FCLayer(reqd_threads, threadblock_size);
  SigmoidBackprop_GPUKernel << < num_threadblocks, threadblock_size >> >
    (d_backprop_derivatives, d_out_xw_act,
     d_fwd_layer_derivatives,
     derivative_matrix_size);
}

//void ComputePrevLayerDerivsFromElemGrads_efficient(float *d_elem_grads,
//                                         float *d_prev_layer_derivatives,
//                                         int input_batch_size,
//                                         int input_neurons,
//                                         int output_neurons) {
//  int num_threadblocks = (input_neurons + 1) * input_batch_size;
//  int sum_stride = 2;
//  int threadblock_size = std::ceilf(std::ceilf((float)output_neurons
//                                               / sum_stride)
//                                    / GPU_WARP_SIZE) * GPU_WARP_SIZE;
//  ComputePrevLayerDerivsFromElemGrads_efficient_GPUKernel <<< num_threadblocks,
//                                              threadblock_size >>>
//                                              (d_elem_grads,
//                                               d_prev_layer_derivatives,
//                                               input_batch_size,
//                                               input_neurons, 
//                                               output_neurons, sum_stride);
//}

//void ComputePrevLayerDerivsFromElemGrads(float *d_elem_grads,
//                                         float *d_prev_layer_derivatives,
//                                         int input_batch_size,
//                                         int input_neurons,
//                                         int output_neurons) {
//  int reqd_threads = (input_neurons + 1) * input_batch_size;
//  int threadblock_size = GPU_WARP_SIZE * GPU_WARP_DISPATCHERS * 2;
//  int num_threadblocks = std::ceilf((float)reqd_threads / threadblock_size);
//  ComputePrevLayerDerivsFromElemGrads_GPUKernel <<< num_threadblocks,
//                                                    threadblock_size >>>
//                                                    (d_elem_grads,
//                                                     d_prev_layer_derivatives,
//                                                     input_batch_size,
//                                                     input_neurons, 
//                                                     output_neurons);
//}