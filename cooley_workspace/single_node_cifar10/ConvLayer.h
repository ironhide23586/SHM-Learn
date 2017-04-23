#pragma once
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>
//#include <curand_kernel.h>
#include <vector>

#include <iostream>
#include <random>
//#include "GlobalInclude.h"

// Computes W - lambda * W^2
void WeightMatrixRegularizeElemWiseConv(float *d_mat_in,
                                        float reg_inp_scalar, int d_mat_size);

void SubtractElemwise_Conv(float *d_mat, float delta, int mat_size);

enum regularizer_type_Conv { L1_Conv, L2_Conv };

class ConvLayer {

public:

  ConvLayer(const cudnnHandle_t &cudnn_handle_arg,
            const cublasHandle_t &cublas_handle_arg,
            int num_images_arg, int input_channels_arg,
            int input_h_arg, int input_w_arg,
            int pad_h_arg, int pad_w_arg,
            int vert_stride_arg, int hor_stride_arg,
            int kernel_h_arg, int kernel_w_arg,
            int feature_maps_arg, float learning_rate_arg = 1e-2f,
            float momentum_arg = 1e-3f,
            float regularization_coeff_arg = 1e-2f,
            regularizer_type_Conv regularizer_arg = L2_Conv,
            float weight_init_mean_arg = 0.0f,
            float weight_init_stddev_arg = 0.5f);
  void LoadData(float *input_data_arg, bool input_data_on_gpu_arg);
  void SetPoolingParams(cudnnPoolingMode_t pool_mode_arg,
                        int pool_height_arg, int pool_width_arg,
                        int pool_vert_stride_arg, int pool_hor_stride_arg,
                        int pool_vert_pad_arg, int pool_hor_pad_arg);
  void Convolve();
  float* GetOutput(void);
  void SetActivationFunc(cudnnActivationMode_t activation_mode,
                         double relu_clip = 1.0f);
  void ComputeLayerGradients(float *d_backprop_derivatives);
  void UpdateWeights(float *d_filter_gradients, float *d_bias_gradients);


  int kernel_h, kernel_w;
  int vert_stride, hor_stride;
  int pad_h, pad_w;
  int feature_maps;

  int input_n, input_c;
  int input_h, input_w;
  int output_n, output_c;
  int output_h, output_w; //These dims correspond to the output of the layer

  int conv_output_n, conv_output_c;
  int conv_output_h, conv_output_w;
  int pool_output_n, pool_output_c;
  int pool_output_h, pool_output_w;

  int x_upscale, y_upscale;

  int pool_width, pool_height;
  int pool_vert_stride, pool_hor_stride;
  int pool_pad_h, pool_pad_w;

  bool pooling_enabled; //Pooling done if set to true
  bool pooling_params_initialized;
  bool input_data_on_gpu; // Set this to true if input data is already on GPU
  bool activation_set;
  bool grads_initialized;
  bool is_input_layer;

  float *d_data, *d_out;

  float *input_data, *h_output;
  float weight_init_mean, weight_init_stddev;
  float learning_rate, momentum, regularization_coeff;
  regularizer_type_Conv regularizer;

  float *d_activation_derivatives;
  float *d_pooling_derivatives;
  float *d_conv_filter_derivatives;
  float *d_conv_bias_derivatives;
  float *d_prev_layer_derivatives;

  float *d_filter_gradients, *d_filter_gradients_prev;
  float *d_filter_gradients_final;
  float *d_bias_gradients, *d_bias_gradients_prev;
  float *d_bias_gradients_final;

  float *d_conv, *d_filt, *d_bias, *d_pool; //d_conv has the output on GPU

  cudnnHandle_t cudnn_handle;
  cublasHandle_t cublas_handle;

  cudaError_t cudaError_stat;
  curandStatus_t curandStatus_stat;
  cudnnStatus_t cudnnStatus_stat;
  cublasStatus_t cublasStatus_stat;

  ~ConvLayer();

private:
  void AllocateGPUMemory(void);
  void InitializeFilters(float mean, float stddev); //Temporary
  float GetRandomNum();
  void InitializeBiases(void); //Temporary
  void InitBackpropVars(void);
  void CustomWeightInitializer(float *d_wt_mat, int wt_mat_sz);
  void Convolve_worker(void);

  float alpha, beta;

  float *grad_swap_tmp;
  float reg_inp_scalar;
  float neg_one_scalar;

  size_t fwd_workspace_size;
  size_t bwd_filter_workspace_size, bwd_data_workspace_size;
  void *d_fwd_workspace;
  void *d_bwd_filter_workspace, *d_bwd_data_workspace;
  bool output_copied_to_host;
  bool d_out_allocated;

  int filter_linear_size, filter_total_size;

  float *d_fwd_derivatives_tmp;

  //convTensor is descriptor to conv output
  cudnnTensorDescriptor_t dataTensor, convTensor, biasTensor, poolTensor;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnPoolingDescriptor_t poolDesc;
  cudnnPoolingMode_t pool_mode;
  cudnnActivationDescriptor_t cudnn_activation_desc;
};
