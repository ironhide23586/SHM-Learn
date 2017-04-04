#include "FCLayer.h"

void print_d_var(float *d_v, int r, int c) {
  std::cout << "*****************************" << std::endl;
  float *h_v = (float *)malloc(sizeof(float) * r * c);
  cudaMemcpy(h_v, d_v, sizeof(float) * r * c, cudaMemcpyDeviceToHost);
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      printf("%f\t", h_v[j + i * c]);
    }
    std::cout << std::endl;
  }
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
        std::cout << h_a[j + i * c] << ", " << h_b[j + i * c] << std::endl;
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
                 int input_batch_size_arg,
                 int input_n_arg, int output_n_arg,
                 bool is_softmax_layer_arg,
                 float learning_rate_arg,
                 float momentum_arg,
                 float regularization_coeff_arg,
                 regularizer_type regularizer_arg,
                 float weight_init_mean_arg,
                 float weight_init_stddev_arg)
    : cudnn_handle(cudnn_handle_arg),
      cublas_handle(cublas_handle_arg),
      input_batch_size(input_batch_size_arg),
      input_neurons(input_n_arg),
      output_neurons(output_n_arg),
      is_softmax_layer(is_softmax_layer_arg),
      learning_rate(learning_rate_arg),
      momentum(momentum_arg),
      regularization_coeff(regularization_coeff_arg),
      regularizer(regularizer_arg),
      weight_init_mean(weight_init_mean_arg),
      weight_init_stddev(weight_init_stddev_arg) {
  weight_matrix_size = (input_neurons + 1) * output_neurons;
  input_data_matrix_size = input_batch_size * (input_neurons + 1);
  weight_matrix_rows = input_neurons + 1;
  weight_matrix_cols = output_neurons;
  alpha = 1.0f;
  beta = 0.0f;
  neg_one_scalar = -1.0f;
  threadBlockSize = GPU_WARP_DISPATCHERS * GPU_WARP_SIZE;
  AllocateGPUMemory();
  InitializeWeightMatrix(weight_init_mean, weight_init_stddev);
  activation_set = false;
  grads_initialized = false;
}

void FCLayer::LoadData(float *input_data_arg, bool input_data_on_gpu_arg) {
  input_data = input_data_arg;
  input_data_on_gpu = input_data_on_gpu_arg;
  if (!input_data_on_gpu) {
    cudaMalloc((void **)&d_data,
               sizeof(float) * input_data_matrix_size);
    cudaMemcpy(d_data, input_data,
               sizeof(float) * (input_data_matrix_size - input_batch_size),
               cudaMemcpyHostToDevice);
  }
  else
    d_data = input_data;
  AddOneVector();
}

void FCLayer::SetActivationFunc(cudnnActivationMode_t activation_mode,
                                double relu_clip) {
  activation_set = true;
  cudnnCreateActivationDescriptor(&cudnn_activation_desc);
  cudnnSetActivationDescriptor(cudnn_activation_desc, activation_mode,
                               CUDNN_PROPAGATE_NAN, relu_clip);
}

void FCLayer::SoftmaxOut() {
  cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, d_out_tensor,
                      d_out, &beta, d_out_tensor, d_out);
}

void FCLayer::ComputeSoftmaxGradientsT(float *h_labels) {
  if (!grads_initialized) {
    cudaError_stat = cudaMalloc((void **)&d_softmax_gradients,
                                sizeof(float) * (input_neurons + 1)
                                * output_neurons);
    cudaError_stat = cudaMalloc((void **)&d_softmax_gradients_T,
                                sizeof(float) * output_neurons
                                * (input_neurons + 1));

    cudaError_stat = cudaMalloc((void **)&d_labels,
                                sizeof(float) * input_batch_size 
                                * output_neurons);
    cudaError_stat = cudaMalloc((void **)&d_out_minus_labels,
                                sizeof(float) * input_batch_size
                                * output_neurons);
    cudaError_stat = cudaMalloc((void **)&d_elem_grads,
                                sizeof(float)
                                * (input_batch_size * (input_neurons + 1))
                                * output_neurons);
    softmax_grad_coeff = 1.0f / input_batch_size;
    grads_initialized = true;
  }
  cudaError_stat = cudaMemcpy(d_labels, h_labels,
                              sizeof(float) * input_batch_size
                              * output_neurons, cudaMemcpyHostToDevice);
  //reinit_vars();

  print_d_var(d_out, input_batch_size, output_neurons);
  print_d_var(d_labels, input_batch_size, output_neurons);

  cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  output_neurons, input_batch_size, &alpha,
                                  d_out, output_neurons, &neg_one_scalar,
                                  d_labels, output_neurons, d_out_minus_labels,
                                  output_neurons);

  print_d_var(d_out_minus_labels, input_batch_size, output_neurons);

  ElemwiseGradCompute(d_data, d_out_minus_labels, d_elem_grads,
                      input_batch_size, input_neurons, output_neurons);

  print_d_var(d_elem_grads, input_batch_size * (input_neurons + 1),
              output_neurons);

  auto st = std::chrono::system_clock::now();
  ComputeGradientsFromElemGrads(d_elem_grads, d_softmax_gradients,
                                learning_rate, momentum, input_batch_size,
                                input_neurons, output_neurons);
  auto et = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(et - st);

  print_d_var(d_softmax_gradients, input_neurons + 1, output_neurons);

  st = std::chrono::system_clock::now();
  cublasStatus_stat = MatMulT(d_data, d_labels, d_softmax_gradients_T,
                              input_batch_size, input_neurons + 1,
                              input_batch_size, output_neurons,
                              softmax_grad_coeff * learning_rate,
                              momentum);
  cublasStatus_stat = MatMulT(d_data, d_out, d_softmax_gradients_T,
                              input_batch_size, input_neurons + 1,
                              input_batch_size, output_neurons,
                              softmax_grad_coeff * learning_rate,
                              -1.0f);

  print_d_var(d_softmax_gradients_T, output_neurons, input_neurons + 1);
  
  d_gradients_T = d_softmax_gradients_T;
  d_gradients = d_softmax_gradients;
}

void FCLayer::UpdateWeights() {
  reg_inp_scalar = 1.0f - regularization_coeff;
  float *d_wm_tmp;
  cudaMalloc((void **)&d_wm_tmp, sizeof(float) * weight_matrix_rows * weight_matrix_cols);
  cudaMemcpy(d_wm_tmp, d_weight_matrix, sizeof(float) * weight_matrix_size, cudaMemcpyDeviceToDevice);
  if (regularizer == L1) {
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    output_neurons, input_neurons + 1,
                                    &neg_one_scalar, d_gradients_T,
                                    input_neurons + 1, &reg_inp_scalar,
                                    d_wm_tmp, output_neurons,
                                    d_wm_tmp, output_neurons);
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    output_neurons, input_neurons + 1,
                                    &neg_one_scalar, d_gradients,
                                    output_neurons, &reg_inp_scalar,
                                    d_weight_matrix, output_neurons,
                                    d_weight_matrix, output_neurons);
    //print_d_var(d_weight_matrix, weight_matrix_rows, weight_matrix_cols);
  }
  else if (regularizer == L2) {
    reg_inp_scalar--;
    WeightMatrixRegularizeElemWise(d_weight_matrix,
                                   reg_inp_scalar, weight_matrix_size);
    
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    output_neurons, input_neurons + 1,
                                    &neg_one_scalar, d_gradients_T,
                                    input_neurons + 1, &alpha,
                                    d_wm_tmp, output_neurons,
                                    d_wm_tmp, output_neurons);
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    output_neurons, input_neurons + 1,
                                    &neg_one_scalar, d_gradients,
                                    output_neurons, &alpha,
                                    d_weight_matrix, output_neurons,
                                    d_weight_matrix, output_neurons);
  }
  bool k = isEqual(d_weight_matrix, d_wm_tmp,
                   weight_matrix_rows, weight_matrix_cols);
  if (k)
    std::cout << "Match!";
  else
    std::cout << "Mismatch :(";
}

void FCLayer::AllocateGPUMemory() {
  cudaError_stat = cudaMalloc((void **)&d_weight_matrix,
                              sizeof(float) * weight_matrix_size);
  cudaError_stat = cudaMalloc((void **)&d_out,
                              sizeof(float) * input_batch_size
                              * (output_neurons + 1));
  cudnnCreateTensorDescriptor(&d_out_tensor);
  cudnnSetTensor4dDescriptor(d_out_tensor, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, input_batch_size,
                             output_neurons, 1, 1);
}

void FCLayer::InitializeWeightMatrix(float mean, float stddev) { //Bias set to 0
  curandGenerator_t rng;
  float *h_tmp_wt = (float *)malloc(sizeof(float));
  CustomWeightInitializer(d_weight_matrix, weight_matrix_size, 0.1f);

  //curandStatus_stat = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  //if (!curandStatus_stat) { //Execute only if curand init succeeds
  //  //print_d_var(d_weight_matrix, 1, 10);
  //  curandStatus_stat = curandGenerateNormal(rng, d_weight_matrix,
  //                                           sizeof(float)
  //                                           * weight_matrix_size,
  //                                           mean, stddev);
  //  //print_d_var(d_weight_matrix, 1, 10);
  //  curandStatus_stat = curandDestroyGenerator(rng);
  //  //cudaError_stat = cudaMemset(d_weight_matrix, 0, sizeof(float)
  //  //                            * weight_matrix_cols);
  //  //print_d_var(d_weight_matrix, 1, 10);
  //}
  //else {
  //  CustomWeightInitializer(d_weight_matrix, weight_matrix_size, 0.1f);
  //}
  //
  //cudaError_stat = cudaMemcpy(h_tmp_wt, d_weight_matrix, sizeof(float) * 2,
  //                            cudaMemcpyDeviceToHost);
  //if (*h_tmp_wt < -1.0f || *h_tmp_wt > 1.0f || !cudaError_stat)
  //  CustomWeightInitializer(d_weight_matrix, weight_matrix_size, 0.1f);
  //print_d_var(d_weight_matrix, 1, 10);
}

void FCLayer::AddOneVector() {
  ReAlignMemory(d_data, input_data_matrix_size, input_neurons,
                input_batch_size,
                std::ceil((float)(input_data_matrix_size - input_batch_size)
                / GPU_WARP_SIZE) * GPU_WARP_SIZE);
  FillOnes(d_data, input_neurons, input_batch_size);
}

void FCLayer::ForwardProp() {
  cublasStatus_t cbt = cublasSgemm_v2(cublas_handle, CUBLAS_OP_T,
                                      CUBLAS_OP_T, input_batch_size, output_neurons,
                                      (input_neurons + 1), &alpha, d_data,
                                      (input_neurons + 1), d_weight_matrix, 
                                      output_neurons, &beta,
                                      d_out, input_batch_size);
  if (!is_softmax_layer && activation_set) {
    cudnnActivationForward(cudnn_handle, cudnn_activation_desc, &alpha,
                           d_out_tensor, d_out, &beta, d_out_tensor,
                           d_out);
  }
  else if (is_softmax_layer) {
    SoftmaxOut();
  }
}

void FCLayer::CustomWeightInitializer(float *d_wt_mat, int wt_mat_sz,
                                      float bias_wt_val) {
  float *h_tmp_wt_mat = (float *)malloc(sizeof(float) * weight_matrix_size);
  for (long int i = 0; i < weight_matrix_size; i++) {
    if (i < weight_matrix_cols)
      h_tmp_wt_mat[i] = bias_wt_val;
    else
      h_tmp_wt_mat[i] = GetRandomNum();
  }
  cudaError_stat = cudaMemcpy(d_weight_matrix, h_tmp_wt_mat,
                              sizeof(float) * weight_matrix_size,
                              cudaMemcpyHostToDevice);
  free(h_tmp_wt_mat);
}

float FCLayer::GetRandomNum() {
  static std::default_random_engine re;
  static std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  return dist(re);
}

void FCLayer::SumColumns(float *d_mat, float *d_out, int rows, int cols) {
  float *d_onevect;
  cudaError_stat = cudaMalloc((void **)&d_onevect,
                              sizeof(float) * rows);
  FloatGPUMemset(d_onevect, rows, 1.0f);
  cublasStatus_stat = cublasSgemv_v2(cublas_handle, CUBLAS_OP_N,
                                     cols, rows,
                                     &alpha, d_mat, cols,
                                     d_onevect, 1, 
                                     &beta, d_out, 1);
  cudaError_stat = cudaFree(d_onevect);
}

cublasStatus_t FCLayer::MatMulT(float *d_A, float *d_B, float *d_C,
                               int rows_A, int cols_A, 
                               int rows_B, int cols_B, float scale_coeff,
                               float prior_coeff) {
  return cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        cols_A, cols_B, rows_A, &scale_coeff,
                        d_A, cols_A, d_B, cols_B, &prior_coeff, d_C,
                        cols_A);
}

void FCLayer::reinit_vars() {
  cudaError_stat = cudaFree(d_softmax_gradients_T);
  cudaError_stat = cudaFree(d_data);
  cudaError_stat = cudaFree(d_out);
  cudaError_stat = cudaFree(d_labels);
  //cudaError_stat = cudaFree(d_gradients_I);
  cudaError_stat = cudaFree(d_weight_matrix);
  //cudaError_stat = cudaFree(d_weight_matrix_squared);

  input_neurons = 3;
  output_neurons = 3;
  input_batch_size = 2;
  softmax_grad_coeff = 1.0f / input_batch_size;
  learning_rate = 1.0f;
  momentum = 0.0f;
  regularization_coeff = 0.0f;

  weight_matrix_rows = input_neurons + 1;
  weight_matrix_cols = output_neurons;

  cudaError_stat = cudaMalloc((void **)&d_softmax_gradients_T,
                              sizeof(float) * output_neurons
                              * (input_neurons + 1));
  cudaError_stat = cudaMalloc((void **)&d_labels,
                              sizeof(float) * input_batch_size
                              * output_neurons);
  cudaError_stat = cudaMalloc((void **)&d_out,
                              sizeof(float) * input_batch_size
                              * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_gradients_I,
  //                            sizeof(float) * output_neurons
  //                            * output_neurons);
  cudaError_stat = cudaMalloc((void **)&d_data,
                              sizeof(float) * input_batch_size
                              * (input_neurons + 1));
  cudaError_stat = cudaMalloc((void **)&d_weight_matrix,
                              sizeof(float) * (input_neurons + 1)
                              * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_weight_matrix_squared,
  //                            sizeof(float) * (input_neurons + 1)
  //                            * output_neurons);

  float h_d[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  float h_o[6] = { 0.1f, 0.5f, 0.4f, 0.7f, 0.2f, 0.1f };
  float h_l[6] = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f };
  //float h_w[12] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  float h_w[12];
  for (int i = 0; i < 12; i++) {
    h_w[i] = (float) i / 10;
  }
  float h_prev_grad[12] = { -0.7,  -0.8, -0.9, -1,
    0.75, 1.1, 1.45, 1.8,
    -0.05, -0.3, -0.55, -0.8 };
  cudaMemcpy(d_data, h_d, sizeof(float) * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_o, sizeof(float) * 6, cudaMemcpyHostToDevice);
  cudaMemcpy(d_labels, h_l, sizeof(float) * 6, cudaMemcpyHostToDevice);
  cudaMemcpy(d_softmax_gradients_T, h_prev_grad, sizeof(float) * 12, cudaMemcpyHostToDevice);
  //InitIdentityMatrix(d_gradients_I, output_neurons);
  cudaMemcpy(d_weight_matrix, h_w, sizeof(float) * 12, cudaMemcpyHostToDevice);
}