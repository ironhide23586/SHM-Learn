#include "ConvLayer.h"

// void print_d_var2(float *d_v, int r, int c) {
//   std::cout << "\n-------------------------" << std::endl;
//   float *h_v = (float *)malloc(sizeof(float) * r * c);
//   cudaMemcpy(h_v, d_v, sizeof(float) * r * c, cudaMemcpyDeviceToHost);
//   for (int i = 0; i < r; i++) {
//     for (int j = 0; j < c; j++) {4
//       std::cout << h_v[j + i * c] << "\t";
//     }
//     std::cout << std::endl;
//   }
// }

void print_d_var2(float *d_v, int r, int c) {
  bool print_elem = false;
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
  // std::cout << std::endl;
  free(h_v);
}

void print_h_var2(float *h_v, int r, int c, bool print_elem = true) {
  std::cout << "-------------------------" << std::endl;
  float mini = h_v[0], maxi = h_v[0];
  float sum = 0.0f;
  int mini_idx = 0, maxi_idx = 0;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if (print_elem)
        std::cout << h_v[j + i * c] << "\t";
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
  // std::cout << std::endl;
}

ConvLayer::ConvLayer(const cudnnHandle_t &cudnn_handle_arg,
                     const cublasHandle_t &cublas_handle_arg,
                     int input_n_arg, int input_c_arg,
                     int input_h_arg, int input_w_arg,
                     int pad_h_arg, int pad_w_arg,
                     int vert_stride_arg, int hor_stride_arg,
                     int kernel_h_arg, int kernel_w_arg,
                     int feature_maps_arg, float learning_rate_arg,
                     float momentum_arg, float regularization_coeff_arg,
                     regularizer_type_Conv regularizer_arg,
                     float weight_init_mean_arg, float weight_init_stddev_arg)
  : cudnn_handle(cudnn_handle_arg),
  cublas_handle(cublas_handle_arg),
  input_n(input_n_arg),
  input_c(input_c_arg),
  input_h(input_h_arg),
  input_w(input_w_arg),
  pad_h(pad_h_arg),
  pad_w(pad_w_arg),
  vert_stride(vert_stride_arg),
  hor_stride(hor_stride_arg),
  kernel_h(kernel_h_arg),
  kernel_w(kernel_w_arg),
  feature_maps(feature_maps_arg),
  learning_rate(learning_rate_arg),
  momentum(momentum_arg),
  regularization_coeff(regularization_coeff_arg),
  regularizer(regularizer_arg),
  weight_init_mean(weight_init_mean_arg),
  weight_init_stddev(weight_init_stddev_arg) {
  cudnnCreateTensorDescriptor(&dataTensor);
  cudnnStatus_stat = cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             input_n, input_c, input_h, input_w);
  cudnnStatus_stat = cudnnCreateConvolutionDescriptor(&convDesc);
  cudnnCreateFilterDescriptor(&filterDesc);
  cudnnCreateTensorDescriptor(&convTensor);
  cudnnCreateTensorDescriptor(&biasTensor);
  x_upscale = 1;
  y_upscale = 1;
  AllocateGPUMemory();
  InitializeFilters(weight_init_mean, weight_init_stddev);
  InitializeBiases();
  activation_set = false;
  output_copied_to_host = false;
  d_out_allocated = false;
  grads_initialized = false;
  is_input_layer = false;
  pooling_params_initialized = false;
  neg_one_scalar = -1.0f;
  
  alpha = 1.0f;
  beta = 0.0f;
}

void ConvLayer::AllocateGPUMemory() {
  cudaMalloc((void **)&d_data,
             input_n * input_c * input_h * input_w * sizeof(float));
  cudnnStatus_stat = cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, vert_stride,
                                                     hor_stride, x_upscale, y_upscale,
                                                     CUDNN_CONVOLUTION);
  cudnnStatus_stat = cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                feature_maps, input_c, kernel_h, kernel_w);
  //no_of_feature_maps, c, h, w; feaure_maps = out_channels
  filter_linear_size = input_c * kernel_h * kernel_w;
  filter_total_size = feature_maps * filter_linear_size;
  cudaMalloc((void **)&d_filt,
             sizeof(float) * feature_maps
             * input_c * kernel_h * kernel_w);
  cudnnStatus_stat = cudnnGetConvolution2dForwardOutputDim(convDesc, dataTensor, filterDesc,
                                                           &output_n, &output_c, &output_h,
                                                           &output_w);

  conv_output_n = output_n;
  conv_output_c = output_c;
  conv_output_h = output_h;
  conv_output_w = output_w;

  cudnnSetTensor4dDescriptor(convTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             output_n, output_c, output_h, output_w);
  cudaMalloc((void **)&d_conv, sizeof(float) * output_n
             * ((output_c * output_h * output_w) + 1)); //+1->future bias
  cudnnGetConvolutionForwardAlgorithm(cudnn_handle, dataTensor, filterDesc,
                                      convDesc, convTensor,
                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                      0, &fwd_algo);
  cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, dataTensor,
                                          filterDesc, convDesc,
                                          convTensor, fwd_algo, &fwd_workspace_size);
  cudaMalloc(&d_fwd_workspace, fwd_workspace_size);
  cudnnSetTensor4dDescriptor(biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             1, output_c, 1, 1);
  cudaMalloc((void **)&d_bias, sizeof(float) * output_c);
}

void ConvLayer::LoadData(float *input_data_arg, bool input_data_on_gpu_arg) {
  input_data = input_data_arg;
  input_data_on_gpu = input_data_on_gpu_arg;
  if (!input_data_on_gpu) {
    //print_h_var2(input_data, input_n, input_c * input_h * input_w, false);
    cudaError_stat = cudaMemcpy(d_data, input_data,
                                sizeof(float) * input_n * input_c
                                * input_h * input_w,
                                cudaMemcpyHostToDevice);
    //std::cout << "Internal cuda mem copy to GPU ---> " << cudaError_stat << std::endl;
  }
  else
    d_data = input_data;
}

void ConvLayer::SetPoolingParams(cudnnPoolingMode_t pool_mode_arg,
                                 int pool_height_arg, int pool_width_arg,
                                 int pool_vert_stride_arg, int pool_hor_stride_arg,
                                 int pool_pad_h_arg, int pool_pad_w_arg) {
  pool_mode = pool_mode_arg;
  pool_vert_stride = pool_vert_stride_arg;
  pool_hor_stride = pool_hor_stride_arg;
  pool_width = pool_width_arg;
  pool_height = pool_height_arg;
  pooling_params_initialized = true;
  pool_pad_h = pool_pad_h_arg;
  pool_pad_w = pool_pad_w_arg;

  cudnnCreatePoolingDescriptor(&poolDesc);
  cudnnCreateTensorDescriptor(&poolTensor);
  cudnnSetPooling2dDescriptor(poolDesc, pool_mode, CUDNN_PROPAGATE_NAN,
                              pool_height, pool_width,
                              pool_pad_h, pool_pad_w,
                              pool_vert_stride, pool_hor_stride);
  cudnnGetPooling2dForwardOutputDim(poolDesc, convTensor,
                                    &output_n, &output_c,
                                    &output_h, &output_w);
  pool_output_n = output_n;
  pool_output_c = output_c;
  pool_output_h = output_h;
  pool_output_w = output_w;
  cudnnSetTensor4dDescriptor(poolTensor, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, output_n, output_c,
                             output_h, output_w);
  cudaMalloc((void **)&d_pool,
             sizeof(float) * output_n * ((output_c * output_h * output_w)
                                         + 1)); //+1 to accomodate bias in future
}

void ConvLayer::InitializeFilters(float mean, float stddev) {
  //curandGenerator_t rng;
  //curandStatus_stat = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  ////print_d_var2(d_filt, feature_maps, input_c * input_h * input_w);
  //curandStatus_stat = curandGenerateNormal(rng, d_filt, sizeof(float) * feature_maps
  //                                         * input_c * kernel_h * kernel_w, mean, stddev);
  ////print_d_var2(d_filt, feature_maps, input_c * kernel_h * kernel_w);
  //curandStatus_stat = curandDestroyGenerator(rng);
  CustomWeightInitializer(d_filt, feature_maps * input_c * kernel_h * kernel_w);
  //cudaMemset(d_filt, 1, sizeof(float) * feature_maps * input_c
  //           * kernel_h * kernel_w);
}

void ConvLayer::InitializeBiases() {
  cudaMemset(d_bias, 0, sizeof(float) * output_c);
}

void ConvLayer::CustomWeightInitializer(float *d_wt_mat, int wt_mat_sz) {
  float *h_tmp_wt_mat = (float *)malloc(sizeof(float) * wt_mat_sz);
  //float wt_avg = 0.0;
  for (long int i = 0; i < wt_mat_sz; i++) {
    h_tmp_wt_mat[i] = GetRandomNum();
    //wt_avg += h_tmp_wt_mat[i];
  }
  //wt_avg /= wt_mat_sz;
  cudaError_stat = cudaMemcpy(d_wt_mat, h_tmp_wt_mat,
                              sizeof(float) * wt_mat_sz,
                              cudaMemcpyHostToDevice);
  //SubtractElemwise_Conv(d_wt_mat, wt_avg, wt_mat_sz);
  free(h_tmp_wt_mat);
}

float ConvLayer::GetRandomNum() {
  static std::default_random_engine re;
  static std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  return dist(re);
}

void ConvLayer::Convolve() {
  Convolve_worker();
}

void ConvLayer::Convolve_worker() {
  cudnnStatus_stat = cudnnConvolutionForward(cudnn_handle, &alpha, dataTensor,
                          d_data, filterDesc, d_filt,
                          convDesc, fwd_algo, d_fwd_workspace,
                          fwd_workspace_size, &beta, convTensor, d_conv);
  std::cout << "cudnn fwd conv ---> " << cudnnStatus_stat << std::endl;
  cudnnAddTensor(cudnn_handle, &alpha, biasTensor,
                 d_bias, &alpha, convTensor, d_conv);
  //d_out = d_conv;
  if (pooling_enabled) {
    cudnnStatus_t st = cudnnPoolingForward(cudnn_handle, poolDesc, &alpha,
                                           convTensor, d_conv, &beta,
                                           poolTensor, d_pool);
    //d_out = d_pool;
  }
  if (activation_set) {
    if (!d_out_allocated) {
      cudaError_stat = cudaMalloc((void **)&d_out, sizeof(float) * output_n
                                  * ((output_c * output_h * output_w) + 1));
      d_out_allocated = true;
    }
    if (pooling_enabled) {
      cudnnActivationForward(cudnn_handle, cudnn_activation_desc, &alpha,
                             poolTensor, d_pool, &beta, poolTensor, d_out);
    }
    else {
      cudnnActivationForward(cudnn_handle, cudnn_activation_desc, &alpha,
                             convTensor, d_conv, &beta, convTensor, d_out);
    }
  }
  else {
    if (pooling_enabled)
      d_out = d_pool;
    else
      d_out = d_conv;
  }
}

float* ConvLayer::GetOutput() {
  output_copied_to_host = true;
  h_output = (float *)malloc(sizeof(float) * output_n
                             * output_c * output_h * output_w);
  float *d_out = pooling_enabled ? d_pool : d_conv;
  cudaMemcpy(h_output, d_out,
             sizeof(float) * output_n * output_c * output_h * output_w,
             cudaMemcpyDeviceToHost);
  return h_output;
}

void ConvLayer::SetActivationFunc(cudnnActivationMode_t activation_mode,
                                  double relu_clip) {
  activation_set = true;
  cudnnCreateActivationDescriptor(&cudnn_activation_desc);
  cudnnSetActivationDescriptor(cudnn_activation_desc, activation_mode,
                               CUDNN_PROPAGATE_NAN, relu_clip);
}

void ConvLayer::ComputeLayerGradients(float *d_backprop_derivatives) {
  if (!grads_initialized) {
    InitBackpropVars();
    grads_initialized = true;
  }
  if (activation_set) {
    if (pooling_enabled) {
      //print_d_var2(d_activation_derivatives, input_n, output_c * output_h * output_w);
      cudnnStatus_stat = cudnnActivationBackward(cudnn_handle,
                                                 cudnn_activation_desc, &alpha,
                                                 poolTensor, d_out, poolTensor,
                                                 d_backprop_derivatives,
                                                 poolTensor, d_pool, &beta,
                                                 poolTensor,
                                                 d_activation_derivatives);
      //print_d_var2(d_activation_derivatives, input_n, output_c * output_h * output_w);
    }
    else {
      cudnnStatus_stat = cudnnActivationBackward(cudnn_handle,
                                                 cudnn_activation_desc, &alpha,
                                                 convTensor, d_out, convTensor,
                                                 d_backprop_derivatives,
                                                 convTensor, d_conv, &beta,
                                                 convTensor,
                                                 d_activation_derivatives);
    }
    d_fwd_derivatives_tmp = d_activation_derivatives;
  }
  else {
    d_fwd_derivatives_tmp = d_backprop_derivatives;
  }

  if (pooling_enabled) {
    //print_d_var2(d_pooling_derivatives, conv_output_n, conv_output_c * conv_output_h * conv_output_w);
    cudnnStatus_stat = cudnnPoolingBackward(cudnn_handle, poolDesc, &alpha,
                                            poolTensor, d_pool, poolTensor,
                                            d_fwd_derivatives_tmp, convTensor,
                                            d_conv, &beta, convTensor,
                                            d_pooling_derivatives);
    //print_d_var2(d_pooling_derivatives, conv_output_n, conv_output_c * conv_output_h * conv_output_w);
    d_fwd_derivatives_tmp = d_pooling_derivatives;
  }

  grad_swap_tmp = d_filter_gradients;
  d_filter_gradients = d_filter_gradients_prev;
  d_filter_gradients_prev = grad_swap_tmp;
  cudnnStatus_stat = cudnnConvolutionBackwardFilter(cudnn_handle, &alpha,
                                                    dataTensor, d_data,
                                                    convTensor,
                                                    d_fwd_derivatives_tmp,
                                                    convDesc, bwd_filter_algo,
                                                    d_bwd_filter_workspace,
                                                    bwd_filter_workspace_size,
                                                    &beta, filterDesc,
                                                    d_filter_gradients);
  std::cout << "cudnn bckwd conv ---> " << cudnnStatus_stat << std::endl;
  grad_swap_tmp = d_bias_gradients;
  d_bias_gradients = d_bias_gradients_prev;
  d_bias_gradients_prev = grad_swap_tmp;
  //print_d_var2(d_fwd_derivatives_tmp, input_n, conv_output_c * conv_output_h * conv_output_w);
  cudnnStatus_stat = cudnnConvolutionBackwardBias(cudnn_handle, &alpha,
                                                  convTensor,
                                                  d_fwd_derivatives_tmp,
                                                  &beta, biasTensor,
                                                  d_bias_gradients);
  //print_d_var2(d_bias_gradients, feature_maps, 1);
  if (!is_input_layer) {
    cudnnStatus_stat = cudnnConvolutionBackwardData(cudnn_handle, &alpha,
                                                    filterDesc, d_filt,
                                                    convTensor,
                                                    d_fwd_derivatives_tmp,
                                                    convDesc, bwd_data_algo,
                                                    d_bwd_data_workspace,
                                                    bwd_data_workspace_size,
                                                    &beta, dataTensor,
                                                    d_prev_layer_derivatives);
  }
}

void ConvLayer::UpdateWeights(float *d_filter_gradients,
                              float *d_bias_gradients) {
  //print_d_var2(d_filt, feature_maps, filter_linear_size);
  reg_inp_scalar = 1.0f - regularization_coeff;
  cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  filter_linear_size, feature_maps,
                                  &learning_rate, d_filter_gradients,
                                  filter_linear_size, &momentum,
                                  d_filter_gradients_prev, filter_linear_size,
                                  d_filter_gradients_final, filter_linear_size);
  cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  feature_maps, 1, &learning_rate,
                                  d_bias_gradients, feature_maps, &momentum,
                                  d_bias_gradients_prev, feature_maps,
                                  d_bias_gradients_final, feature_maps);
  if (regularizer == L1_Conv) {
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    filter_linear_size, feature_maps,
                                    &neg_one_scalar, d_filter_gradients_final,
                                    filter_linear_size, &reg_inp_scalar,
                                    d_filt, filter_linear_size,
                                    d_filt, filter_linear_size);
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    feature_maps, 1,
                                    &neg_one_scalar, d_bias_gradients_final,
                                    feature_maps, &reg_inp_scalar,
                                    d_bias, feature_maps,
                                    d_bias, feature_maps);
  }
  else if (regularizer == L2_Conv) {
    reg_inp_scalar--;
    WeightMatrixRegularizeElemWiseConv(d_filt, reg_inp_scalar,
                                       filter_total_size);
    WeightMatrixRegularizeElemWiseConv(d_bias, reg_inp_scalar,
                                       feature_maps);
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    filter_linear_size, feature_maps,
                                    &neg_one_scalar, d_filter_gradients_final,
                                    filter_linear_size, &alpha,
                                    d_filt, filter_linear_size,
                                    d_filt, filter_linear_size);
    cublasStatus_stat = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    feature_maps, 1,
                                    &neg_one_scalar, d_bias_gradients_final,
                                    feature_maps, &alpha,
                                    d_bias, feature_maps,
                                    d_bias, feature_maps);
  }
  //print_d_var2(d_filt, feature_maps, filter_linear_size);
}

void ConvLayer::InitBackpropVars() {
  if (activation_set) {
    cudaError_stat = cudaMalloc((void **)&d_activation_derivatives,
                                sizeof(float) * output_n * output_c
                                * output_w * output_h);
  }
  if (pooling_enabled) {
    cudaError_stat = cudaMalloc((void **)&d_pooling_derivatives,
                                sizeof(float) * conv_output_n * conv_output_c
                                * conv_output_h * conv_output_w);
  }
  cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle, dataTensor,
                                             convTensor, convDesc,
                                             filterDesc,
                                             CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                             0, &bwd_filter_algo);
  cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle, dataTensor,
                                                 convTensor, convDesc,
                                                 filterDesc, bwd_filter_algo,
                                                 &bwd_filter_workspace_size);
  cudaError_stat = cudaMalloc((void **)&d_bwd_filter_workspace,
                              bwd_filter_workspace_size);

  cudaError_stat = cudaMalloc((void **)&d_filter_gradients,
                              sizeof(float) * feature_maps
                              * filter_linear_size);
  cudaError_stat = cudaMalloc((void **)&d_filter_gradients_prev,
                              sizeof(float) * feature_maps
                              * filter_linear_size);
  cudaError_stat = cudaMalloc((void **)&d_filter_gradients_final,
                              sizeof(float) * feature_maps
                              * filter_linear_size);

  cudaError_stat = cudaMalloc((void **)&d_bias_gradients,
                              sizeof(float) * feature_maps);
  cudaError_stat = cudaMalloc((void **)&d_bias_gradients_prev,
                              sizeof(float) * feature_maps);
  cudaError_stat = cudaMalloc((void **)&d_bias_gradients_final,
                              sizeof(float) * feature_maps);

  cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle, filterDesc,
                                           convTensor, convDesc, dataTensor,
                                           CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                           0, &bwd_data_algo);
  cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle, filterDesc,
                                               convTensor, convDesc,
                                               dataTensor, bwd_data_algo,
                                               &bwd_data_workspace_size);
  cudaError_stat = cudaMalloc((void **)&d_bwd_data_workspace,
                              bwd_data_workspace_size);
  if (!is_input_layer) {
    cudaError_stat = cudaMalloc((void **)&d_prev_layer_derivatives,
                                sizeof(float) * input_n * input_c
                                * input_h * input_w);
  }
}

ConvLayer::~ConvLayer() {
  cudaFree(d_data);
  cudaFree(d_conv);
  cudaFree(d_filt);
  cudaFree(d_fwd_workspace);
  if (!input_data_on_gpu)
    free(input_data);
  if (output_copied_to_host)
    free(h_output);
}