#include "FCLayer.h"

void print_d_var(float *d_v, int r, int c, bool print_elem = true) {
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

bool isEqual(float *d_a, float *d_b, int r, int c) {
  float *h_a = (float *)malloc(sizeof(float) * r * c);
  cudaMemcpy(h_a, d_a, sizeof(float) * r * c, cudaMemcpyDeviceToHost);
  float *h_b = (float *)malloc(sizeof(float) * r * c);
  cudaMemcpy(h_b, d_b, sizeof(float) * r * c, cudaMemcpyDeviceToHost);
  bool ret = true;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if ((h_a[j + i * c] - h_b[j + i * c]) > 1e-2f
          || (h_b[j + i * c] - h_a[j + i * c]) > 1e-2f) {
        std::cout << h_a[j + i * c] << ", " << h_b[j + i * c]
          << "(" << i << ", " << j << ")" << std::endl;
        ret = false;
      }
    }
  }
  free(h_a);
  free(h_b);
  return ret;
}

bool isEqualT(float *d_a, float *d_b, int r_a, int c_a) {
  float *h_a = (float *)malloc(sizeof(float) * r_a * c_a);
  cudaMemcpy(h_a, d_a, sizeof(float) * r_a * c_a, cudaMemcpyDeviceToHost);
  float *h_b = (float *)malloc(sizeof(float) * r_a * c_a);
  cudaMemcpy(h_b, d_b, sizeof(float) * r_a * c_a, cudaMemcpyDeviceToHost);
  bool ret = true;
  for (int i = 0; i < r_a; i++) {
    for (int j = 0; j < c_a; j++) {
      if ((h_a[j + i * c_a] - h_b[i + j * r_a]) > 1e-2f
          || (h_b[i + j * r_a] - h_a[j + i * c_a]) > 1e-2f) {
        std::cout << h_a[j + i * c_a] << ", " << h_b[i + j * r_a] << std::endl;
        ret = false;
      }
    }
  }
  free(h_a);
  free(h_b);
  return ret;
}

FCLayer::FCLayer(const cudnnHandle_t &cudnn_handle_arg,
                 const cublasHandle_t &cublas_handle_arg,
                 const cudaDeviceProp &cuda_device_prop_arg,
                 int input_batch_size_arg,
                 int input_n_arg, int output_n_arg,
                 bool is_softmax_layer_arg,
                 float learning_rate_arg,
                 float momentum_arg,
                 float regularization_coeff_arg,
                 float weight_init_mean_arg,
                 float weight_init_stddev_arg,
                 regularizer_type_FC regularizer_arg)
  : cudnn_handle(cudnn_handle_arg),
  cublas_handle(cublas_handle_arg),
  cuda_device_prop(cuda_device_prop_arg),
  input_batch_size(input_batch_size_arg),
  input_neurons(input_n_arg),
  output_neurons(output_n_arg),
  is_softmax_layer(is_softmax_layer_arg),
  learning_rate(learning_rate_arg),
  momentum(momentum_arg),
  regularization_coeff(regularization_coeff_arg),
  weight_init_mean(weight_init_mean_arg),
  weight_init_stddev(weight_init_stddev_arg),
  regularizer(regularizer_arg) {
  is_input_layer = false;
  weight_matrix_size = (input_neurons + 1) * output_neurons;
  input_data_matrix_size = input_batch_size * (input_neurons + 1);
  weight_matrix_rows = input_neurons + 1;
  weight_matrix_cols = output_neurons;
  alpha = 1.0f;
  beta = 0.0f;
  neg_one_scalar = -1.0f;
  threadBlockSize = GPU_WARP_DISPATCHERS * GPU_WARP_SIZE;
  AllocateGPUMemory();
  InitializeWeightMatrix();
  activation_set = false;
  grads_initialized = false;
  d_data_allocated = false;
  max_img_side_dim_3_channel = sqrt(cuda_device_prop.sharedMemPerBlock
                                    / (sizeof(float) * 3));
  neg_learning_rate = -learning_rate;
}


void FCLayer::LoadData(float *input_data_arg, bool input_data_on_gpu_arg) {
  input_data = input_data_arg;
  input_data_on_gpu = input_data_on_gpu_arg;

  if (!input_data_on_gpu) {
    if (!d_data_allocated) {
      CudaSafeCall(cudaMalloc((void **)&d_data, sizeof(float)
                              * input_data_matrix_size));
      d_data_allocated = true;
    }
    CudaSafeCall(cudaMemcpy(d_data, input_data,
                            sizeof(float)
                            * (input_data_matrix_size - input_batch_size),
                            cudaMemcpyHostToDevice));
  }
  else
    d_data = input_data;
  
  AddOneVector_GPU(d_data, input_batch_size, input_neurons);
  CudaCheckError();
}

void FCLayer::SetActivationFunc(cudnnActivationMode_t activation_mode_arg,
                                double relu_clip) {
  activation_set = true;
  activation_mode = activation_mode_arg;
  CudnnSafeCall(cudnnCreateActivationDescriptor(&cudnn_activation_desc));
  CudnnSafeCall(cudnnSetActivationDescriptor(cudnn_activation_desc, activation_mode_arg,
                                             CUDNN_PROPAGATE_NAN, relu_clip));
  CudaSafeCall(cudaMalloc((void **)&d_out_xw_act,
                          sizeof(float) * input_batch_size
                          * (output_neurons + 1)));
  InitializeWeightMatrix();
  relu_activation_clip = relu_clip;
}

void FCLayer::SoftmaxOut() {
  CudnnSafeCall(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, d_out_tensor,
                                    d_out_xw_act, &beta, d_out_tensor, d_out));
}

void FCLayer::ComputeSoftmaxGradients(float *h_pinned_labels) {
  if (!grads_initialized) {
    InitBackpropVars();
    grads_initialized = true;
  }
  d_labels = h_pinned_labels;

  //print_d_var(d_labels, input_batch_size, output_neurons);
  //print_d_var(d_out, input_batch_size, output_neurons);
  CublasSafeCall(cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             output_neurons, input_batch_size,
                             &softmax_grad_coeff, d_out, output_neurons,
                             &neg_softmax_grad_coeff, d_labels,
                             output_neurons, d_out_minus_labels,
                             output_neurons));
  //print_d_var(d_out_minus_labels, input_batch_size, output_neurons);
  //auto st = std::chrono::high_resolution_clock::now();
  ComputeLayerGradients(d_out_minus_labels);
  //auto et = std::chrono::high_resolution_clock::now();
  //float dur = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(et - st).count() * 1e-9f;
}

void FCLayer::ComputeLayerGradients(float *d_backprop_derivatives) {
  if (!grads_initialized) {
    InitBackpropVars();
    grads_initialized = true;
  }

  if (!is_softmax_layer) {
    //ReAlignMemory_ShiftLeft_CPU(d_out_xw_act, input_batch_size, output_neurons + 1);
    ReAlignMemory_ShiftLeft(d_out_xw_act, d_shift_helper,
                            input_batch_size, output_neurons + 1,
                            cuda_device_prop.maxThreadsPerBlock);
    CudaCheckError();
  }
  if (activation_set) {
    //print_d_var(d_backprop_derivatives, input_batch_size, output_neurons, false);
    //print_d_var(d_out_xw_act, input_batch_size, output_neurons);
    //cudnnStatus_stat = cudnnActivationBackward(cudnn_handle,
    //                                           cudnn_activation_desc, &alpha,
    //                                           d_out_tensor, d_out_xw_act,
    //                                           d_out_tensor,
    //                                           d_backprop_derivatives,
    //                                           d_out_tensor, d_out_xw, &beta,
    //                                           d_out_tensor,
    //                                           d_fwd_layer_derivatives);
    if (activation_mode == CUDNN_ACTIVATION_RELU) {
      ReluBackprop(d_backprop_derivatives, d_out_xw_act,
                   d_fwd_layer_derivatives,
                   relu_activation_clip, input_batch_size * output_neurons);
      CudaCheckError();
    }
    else if (activation_mode == CUDNN_ACTIVATION_SIGMOID) {
      SigmoidBackprop(d_backprop_derivatives, d_out_xw_act,
                      d_fwd_layer_derivatives,
                      input_batch_size * output_neurons);
      CudaCheckError();
    }
  }
  else {
    d_fwd_layer_derivatives = d_backprop_derivatives;
  }
  //print_d_var(d_fwd_layer_derivatives, input_batch_size, output_neurons);
  //print_d_var(d_data, input_batch_size, input_neurons + 1);
  CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_N,
                                CUBLAS_OP_T, output_neurons,
                                input_neurons + 1, input_batch_size,
                                &learning_rate, d_fwd_layer_derivatives,
                                output_neurons,
                                d_data,
                                input_neurons + 1,
                                &momentum, d_gradients,
                                output_neurons));
  //print_d_var(d_gradients, input_neurons + 1, output_neurons);
  if (!is_input_layer) {
    //print_d_var(d_gradients, input_neurons + 1, output_neurons);
    //print_d_var(d_data, input_batch_size, input_neurons + 1);
    ComputePrevLayerDerivatives(d_backprop_derivatives);
  }
}

void FCLayer::ComputePrevLayerDerivatives(float *d_fwd_derivatives) {
  CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                input_neurons + 1, input_batch_size,
                                output_neurons, &alpha, d_weight_matrix,
                                output_neurons, d_fwd_derivatives,
                                output_neurons, &beta,
                                d_prev_layer_derivatives,
                                input_neurons + 1));
  ReAlignMemory_ShiftLeft(d_prev_layer_derivatives, d_shift_helper,
                          input_batch_size, input_neurons + 1,
                          cuda_device_prop.maxThreadsPerBlock);
  //Removing irrelevant "bias backprop grads" or the first column of the
  //matrix resulting from the previous step
  CudaCheckError();
}

void FCLayer::UpdateWeights(float *d_update_gradients) {
  reg_inp_scalar = 1.0f - ((learning_rate * regularization_coeff));
  if (regularizer == L1) { //FIX THIS
    CublasSafeCall(cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               output_neurons, input_neurons + 1,
                               &neg_one_scalar, d_update_gradients,
                               output_neurons, &reg_inp_scalar,
                               d_weight_matrix, output_neurons,
                               d_weight_matrix, output_neurons));
  }
  else if (regularizer == L2) {
    WeightMatrixRegularizeElemWise(d_weight_matrix, weight_matrix_cols,
                                   reg_inp_scalar, weight_matrix_size);
    CudaCheckError();
    CublasSafeCall(cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               output_neurons, input_neurons + 1,
                               &neg_one_scalar, d_update_gradients,
                               output_neurons, &alpha,
                               d_weight_matrix, output_neurons,
                               d_weight_matrix, output_neurons));
  }
}

void FCLayer::AllocateGPUMemory() {
  CudaSafeCall(cudaMalloc((void **)&d_weight_matrix,
                          sizeof(float) * weight_matrix_size));
  CudaSafeCall(cudaMalloc((void **)&d_shift_helper,
                          sizeof(float) * input_batch_size
                          * (input_batch_size - 1) / 2));
  if (is_softmax_layer) {
    CudaSafeCall(cudaMalloc((void **)&d_out,
                            sizeof(float) * input_batch_size
                            * (output_neurons + 1)));
  }
  CudaSafeCall(cudaMalloc((void **)&d_out_xw,
                          sizeof(float) * input_batch_size
                          * (output_neurons + 1)));
  CudnnSafeCall(cudnnCreateTensorDescriptor(&d_out_tensor));
  CudnnSafeCall(cudnnSetTensor4dDescriptor(d_out_tensor, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, input_batch_size,
                                           output_neurons, 1, 1));
  if (is_input_layer) {
    CudaSafeCall(cudaMalloc((void **)&d_data, sizeof(float)
                            * input_data_matrix_size));
    d_data_allocated = true;
  }
}

void FCLayer::InitializeWeightMatrix() { //Bias set to 0
  CustomWeightInitializer(d_weight_matrix, weight_matrix_size, 0.0f);
  CudaCheckError();
}

void FCLayer::AddOneVector_GPU(float *d_mat, int rows, int cols) {
  
  ReAlignMemory_ShiftRight(d_mat, d_shift_helper, rows, cols,
                           cuda_device_prop.maxThreadsPerBlock);
  CudaCheckError();
}

void FCLayer::AddOneVector_CPU(float *d_mat, int rows, int cols) {
  int sz = rows * (cols + 1);

  float *tmp0 = (float *)malloc(sizeof(float) * rows * cols);
  float *tmp1 = (float *)malloc(sizeof(float) * sz);

  CudaSafeCall(cudaMemcpy(tmp0, d_mat, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));

  for (int i = 0; i < rows; i++) {
    tmp1[i * (cols + 1)] = 1.0f;
    for (int j = 0; j < cols; j++) {
      tmp1[(j + 1) + i * (cols + 1)] = tmp0[j + i * cols];
    }
  }

  CudaSafeCall(cudaMemcpy(d_mat, tmp1, sizeof(float) * sz, cudaMemcpyHostToDevice));
  free(tmp0);
  free(tmp1);
}

void FCLayer::ForwardProp() {
  //print_d_var(d_data, input_batch_size, input_neurons + 1, false);
  CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_N,
                                CUBLAS_OP_N, output_neurons, input_batch_size,
                                (input_neurons + 1), &alpha, d_weight_matrix,
                                output_neurons, d_data,
                                (input_neurons + 1), &beta,
                                d_out_xw, output_neurons));
  if (!is_softmax_layer && !activation_set) { //Defaults to 0 clipped ReLU
    SetActivationFunc(CUDNN_ACTIVATION_RELU);
  }
  if (activation_set) {
    //print_d_var(d_out_xw, input_batch_size, output_neurons, false);
    //print_d_var(d_out_xw_act, input_batch_size, output_neurons, false);
    CudnnSafeCall(cudnnActivationForward(cudnn_handle, cudnn_activation_desc, &alpha,
                                         d_out_tensor, d_out_xw, &beta, d_out_tensor,
                                         d_out_xw_act));
    //print_d_var(d_out_xw_act, input_batch_size, output_neurons, false);
  }
  else {
    d_out_xw_act = d_out_xw;
  }
  if (is_softmax_layer) {
    SoftmaxOut();
  }
  else {
    d_out = d_out_xw_act;
  }
}

void FCLayer::CustomWeightInitializer(float *d_wt_mat, int wt_mat_sz,
                                      float bias_wt_val) {
  float *h_tmp_wt_mat = (float *)malloc(sizeof(float) * weight_matrix_size);
  float wt_avg = 0.0;
  float coeff;

  if (activation_mode == CUDNN_ACTIVATION_RELU) {
    coeff = std::sqrt(2.0 / (weight_matrix_size - weight_matrix_cols));
  }
  else {
    coeff = 1.0 / std::sqrt(weight_matrix_size - weight_matrix_cols);
  }
  coeff = 1.0;
  for (long int i = 0; i < weight_matrix_size; i++) {
    if (i < weight_matrix_cols)
      h_tmp_wt_mat[i] = bias_wt_val;
    else {
      h_tmp_wt_mat[i] = GetRandomNum(weight_init_mean, weight_init_stddev)
                        * coeff;
      wt_avg += h_tmp_wt_mat[i];
    }
  }

  wt_avg /= (weight_matrix_size - weight_matrix_cols);
  for (long int i = 0; i < weight_matrix_cols; i++) {
    h_tmp_wt_mat[i] += wt_avg;
  }
  CudaSafeCall(cudaMemcpy(d_weight_matrix, h_tmp_wt_mat,
                          sizeof(float) * weight_matrix_size,
                          cudaMemcpyHostToDevice));
  SubtractElemwise(d_weight_matrix, wt_avg, weight_matrix_size);
  //print_d_var(d_weight_matrix, weight_matrix_rows, weight_matrix_cols, false);
  CudaCheckError();
  free(h_tmp_wt_mat);
}

// void FCLayer::CustomWeightInitializer(float *d_wt_mat, int wt_mat_sz,
//                                       float bias_wt_val) {
//   float *h_tmp_wt_mat = (float *)malloc(sizeof(float) * weight_matrix_size);
//   float wt_avg = 0.0;
//   for (long int i = 0; i < weight_matrix_size; i++) {
//     if (i < weight_matrix_cols)
//       h_tmp_wt_mat[i] = bias_wt_val;
//     else {
//       h_tmp_wt_mat[i] = GetRandomNum(weight_init_mean, weight_init_stddev);
//       wt_avg += h_tmp_wt_mat[i];
//     }
//   }
//   wt_avg /= (weight_matrix_size - weight_matrix_cols);
//   for (long int i = 0; i < weight_matrix_cols; i++) {
//     h_tmp_wt_mat[i] += wt_avg;
//   }
//   CudaSafeCall(cudaMemcpy(d_weight_matrix, h_tmp_wt_mat,
//                           sizeof(float) * weight_matrix_size,
//                           cudaMemcpyHostToDevice));
//   SubtractElemwise(d_weight_matrix, wt_avg, weight_matrix_size);
//   CudaCheckError();
//   free(h_tmp_wt_mat);
// }


float FCLayer::GetRandomNum(float mean, float stddev) {
  static std::default_random_engine re;
  //static std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  static std::normal_distribution<float> dist(mean, stddev);
  return dist(re);
}

void FCLayer::SumColumns(float *d_mat, float *d_out, int rows, int cols) {
  float *d_onevect;
  CudaSafeCall(cudaMalloc((void **)&d_onevect,
                          sizeof(float) * rows));
  FloatGPUMemset(d_onevect, rows, 1.0f);
  CudaCheckError();
  CublasSafeCall(cublasSgemv_v2(cublas_handle, CUBLAS_OP_N,
                                cols, rows,
                                &alpha, d_mat, cols,
                                d_onevect, 1,
                                &beta, d_out, 1));
  CudaSafeCall(cudaFree(d_onevect));
}

void FCLayer::InitBackpropVars() {
  if (!is_input_layer) {
    CudaSafeCall(cudaMalloc((void **)&d_prev_layer_derivatives,
                            sizeof(float) * (input_neurons + 1)
                            * input_batch_size));
  }
  if (is_softmax_layer) {

    CudaSafeCall(cudaMalloc((void **)&d_labels,
                            sizeof(float) * input_batch_size
                            * output_neurons));
    CudaSafeCall(cudaMalloc((void **)&d_out_minus_labels,
                            sizeof(float) * input_batch_size
                            * output_neurons));
    CudaSafeCall(cudaMalloc((void **)&d_softmax_derivatives,
                            sizeof(float) * input_batch_size
                            * output_neurons));
    softmax_grad_coeff = 1.0f / input_batch_size;
    neg_softmax_grad_coeff = -softmax_grad_coeff;
  }
  if (activation_set) {
    CudaSafeCall(cudaMalloc((void **)&d_fwd_layer_derivatives,
                            sizeof(float) * input_batch_size
                            * output_neurons));
  }
  CudaSafeCall(cudaMalloc((void **)&d_gradients,
                          sizeof(float) * (input_neurons + 1)
                          * output_neurons));
  grads_initialized = true;
}