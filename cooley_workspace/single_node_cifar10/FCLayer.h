#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>

#include <iostream>
#include <random>
#include <chrono>

#define GPU_WARP_DISPATCHERS 2
#define GPU_WARP_SIZE 32

#define __cudaSafeCall __cudaSafeCall_FC
#define __cudnnSafeCall __cudnnSafeCall_FC
#define __cublasSafeCall __cublasSafeCall_FC
#define __cudaCheckError __cudaCheckError_FC
#define cublasGetErrorString cublasGetErrorString_FC

#include "err_check.h"

using namespace std;

//All vectors assumed to be column vectors
void ReAlignMemory_ShiftLeft(float *d_mat, float *d_helper,
                             int rows, int cols,
                             int max_threadblock_size);

void ReAlignMemory_ShiftRight(float *d_mat, float *d_helper,
                            int rows, int cols, int max_threadblock_size);

void ReluBackprop(float *d_backprop_derivatives, float *d_out_xw_act,
                float *d_fwd_layer_derivatives, float relu_clip,
                int derivative_matrix_size);

void SigmoidBackprop(float *d_backprop_derivatives, float *d_out_xw_act,
                   float *d_fwd_layer_derivatives,
                   int derivative_matrix_size);

void SubtractElemwise(float *d_mat, float delta, int mat_size);

// Computes W - lambda * W^2
// Computes (1 - lr * reg_coeff / batch_size) * W
void WeightMatrixRegularizeElemWise(float *d_mat_in, int d_mat_cols,
                                    float reg_inp_scalar, int d_mat_size);

void FloatGPUMemset(float *d_array_arg,
                  int array_size_arg, float val);

enum regularizer_type_FC { L1, L2 };


//MOMENTUM IMPLEMENTATION FAULTY; momentum multiplied with
//prev_gradient*learningrate
//Fully Connected Layer
class FCLayer {
public:
  FCLayer(const cudnnHandle_t &cudnn_handle_arg,
          const cublasHandle_t &cublas_handle_arg,
          const cudaDeviceProp &cuda_device_prop_arg,
          int input_batch_size_arg, int input_n_arg,
          int output_n_arg, bool is_softmax_layer_arg = false,
          float learning_rate_arg = 1e-2f,
          float momentum_arg = 1e-3f,
          float regularization_coeff_arg = 1e-3f,
          regularizer_type_FC regularizer_arg = L2,
          float weight_init_mean_arg = 0.0f,
          float weight_init_stddev_arg = 1.0f);
  void LoadData(float *input_data_arg, bool input_data_on_gpu_arg);

  // This automatically applies softmax if is_softmax_layer is true
  void ForwardProp();

  // Do NOT set activation if softmax layer
  void SetActivationFunc(cudnnActivationMode_t activation_mode_arg,
                         double relu_clip = 0.0f);
  void SoftmaxOut();

  // Returns matrix with dimension (input_neurons + 1, output_neurons)
  void ComputeSoftmaxGradients(float *h_labels);

  void ComputeLayerGradients(float *d_backprop_derivatives);

  //Input gradient matrix of dimensions (input_neurons + 1) * output_neurons
  void UpdateWeights(float *d_update_gradients);

  void InitBackpropVars();

  regularizer_type_FC regularizer;

  int threadBlockSize;
  int max_img_side_dim_3_channel;
  int output_neurons, input_neurons; //output_neurons->#Neurons,
                                     //input_neurons->#Neurons in previous layer
  int input_batch_size, input_data_matrix_size;
  int weight_matrix_size, weight_matrix_rows, weight_matrix_cols;
  float *input_data, *h_out;
  float weight_init_mean, weight_init_stddev;
  float learning_rate, momentum, regularization_coeff;
  bool input_data_on_gpu, activation_set;
  bool grads_initialized;
  bool is_softmax_layer;
  bool is_input_layer;

  float *d_data, *d_out, *d_labels, *d_out_xw, *d_out_xw_act;
  float *d_out_pre_softmax;
  float *d_softmax_weight_matrix;
  float *d_weight_matrix; //#Rows = input_neurons + 1, #Cols = output_neurons
  float *d_gradients_T, *d_gradients;

  //REMOVE 1st column & change dim to input_batch_size * input_neurons
  float *d_prev_layer_derivatives; //input_batch_size * (input_neurons + 1); These are
                                   //propagated backwards
  float *d_fwd_layer_derivatives; //input_batch_size * output_neurons
  float *d_softmax_derivatives;

  float relu_activation_clip;

  cublasHandle_t cublas_handle;
  cudnnHandle_t cudnn_handle;
  cudnnActivationDescriptor_t cudnn_activation_desc;
  cudnnActivationMode_t activation_mode;
  cudaDeviceProp cuda_device_prop;


private:
  void AllocateGPUMemory();
  void InitializeWeightMatrix(float mean, float stddev);
  void AddOneVector_CPU(float *d_mat, int rows, int cols); //Adds a column of 1's to input data matrix
  void AddOneVector_GPU(float *d_mat, int rows, int cols); //Adds a column of 1's to input data matrix
  void CustomWeightInitializer(float *d_wt_mat, int len, float val);
  float GetRandomNum();
  void SumColumns(float *d_mat, float *d_out, int rows, int cols);


  // Input -> (input_batch_size * output_neurons)
  void ComputePrevLayerDerivatives(float *d_fwd_derivatives);

  void reinit_vars(); //REMOVE THIS LATER; used to set specific vars to check for 
                      //correct results by hand

                      // Returns the transpose of (scale_coeff*A.T*B + prior_coeff*C); 
                      // Python equivalent: C = (scale_coeff*np.dot(A.T, B) + prior_coeff*C).T
  cublasStatus_t MatMulT(float *d_A, float *d_B, float *d_C,
                         int rows_A, int cols_A,
                         int rows_B, int cols_B, float scale_coeff = 1.0f,
                         float prior_coeff = 0.0f);

  float alpha, beta;
  float *d_shift_helper;
  //float *d_softmax_gradients;
  //float *d_softmax_gradients_T; //Contains weight grads for softmax backprop
  float *d_out_minus_labels; //contains d_out - d_labels elem-wise
  float *d_elem_grads; //contains weight gradients for each example in batch
                       //It's columns are summed to get gradients
                       //float *d_weight_matrix_squared;
  float neg_one_scalar; // contains -1
  float neg_learning_rate;
  float *d_onevect;
  float reg_inp_scalar;
  float softmax_grad_coeff, neg_softmax_grad_coeff;

  bool d_data_allocated;

  cudnnTensorDescriptor_t d_out_tensor;

  cudaError_t cudaError_stat;
  curandStatus_t curandStatus_stat;
  cudnnStatus_t cudnnStatus_stat;
  cublasStatus_t cublasStatus_stat;
};