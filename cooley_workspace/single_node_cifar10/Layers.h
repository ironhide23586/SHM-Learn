#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>

#include <iostream>
#include <random>
#include <chrono>

//#include "ErrorChecking_Debug.h"

using namespace std;

#define CudaSafeCall( err )  __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudnnSafeCall( err ) __cudnnSafeCall( err, __FILE__, __LINE__ )
#define CublasSafeCall( err ) __cublasSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

#define GPU_WARP_DISPATCHERS 2
#define GPU_WARP_SIZE 32



namespace NeuralNet {

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}

inline void __cudaSafeCall(cudaError err,
                                         const char *file,
                                         const int line) {
  #ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString(err));
      exit(-1);
    }
  #endif
  return;
}

inline void __cudnnSafeCall(cudnnStatus_t err,
                                          const char *file,
                                          const int line) {
  #ifdef CUDA_ERROR_CHECK
    if (CUDNN_STATUS_SUCCESS != err) {
      fprintf(stderr, "cudnnSafeCall() failed at %s:%i : %s\n",
              file, line, cudnnGetErrorString(err));
    exit(-1);
  }
  #endif
  return;
}

inline void __cublasSafeCall(cublasStatus_t err,
                                           const char *file,
                                           const int line) {
  #ifdef CUDA_ERROR_CHECK
    if (CUBLAS_STATUS_SUCCESS != err) {
      fprintf(stderr, "cublasSafeCall() failed at %s:%i : %s\n",
              file, line, cublasGetErrorString(err));
      exit(-1);
    }
  #endif
  return;
}

inline void __cudaCheckError(const char *file,
                                           const int line) {
  #ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString(err));
      exit( -1 );
    }
  #endif
  return;
}
  
  // // Computes W - lambda * W^2
  // void WeightMatrixRegularizeElemWiseConv(float *d_mat_in,
  //                                         float reg_inp_scalar, int d_mat_size);

  // //void SubtractElemwise_Conv(float *d_mat, float delta, int mat_size);

  // enum regularizer_type_Conv { L1_Conv, L2_Conv };

  // class ConvLayer {

  // public:

  //   ConvLayer(const cudnnHandle_t &cudnn_handle_arg,
  //             const cublasHandle_t &cublas_handle_arg,
  //             int num_images_arg, int input_channels_arg,
  //             int input_h_arg, int input_w_arg,
  //             int pad_h_arg, int pad_w_arg,
  //             int vert_stride_arg, int hor_stride_arg,
  //             int kernel_h_arg, int kernel_w_arg,
  //             int feature_maps_arg, float learning_rate_arg = 1e-2f,
  //             float momentum_arg = 1e-3f,
  //             float regularization_coeff_arg = 1e-2f,
  //             regularizer_type_Conv regularizer_arg = L2_Conv,
  //             float weight_init_mean_arg = 0.0f,
  //             float weight_init_stddev_arg = 0.5f);
  //   void LoadData(float *input_data_arg, bool input_data_on_gpu_arg);
  //   void SetPoolingParams(cudnnPoolingMode_t pool_mode_arg,
  //                         int pool_height_arg, int pool_width_arg,
  //                         int pool_vert_stride_arg, int pool_hor_stride_arg,
  //                         int pool_vert_pad_arg, int pool_hor_pad_arg);
  //   void Convolve();
  //   float* GetOutput(void);
  //   void SetActivationFunc(cudnnActivationMode_t activation_mode,
  //                          double relu_clip = 1.0f);
  //   void ComputeLayerGradients(float *d_backprop_derivatives);
  //   void UpdateWeights(float *d_filter_gradients, float *d_bias_gradients);


  //   int kernel_h, kernel_w;
  //   int vert_stride, hor_stride;
  //   int pad_h, pad_w;
  //   int feature_maps;

  //   int input_n, input_c;
  //   int input_h, input_w;
  //   int output_n, output_c;
  //   int output_h, output_w; //These dims correspond to the output of the layer

  //   int conv_output_n, conv_output_c;
  //   int conv_output_h, conv_output_w;
  //   int pool_output_n, pool_output_c;
  //   int pool_output_h, pool_output_w;

  //   int x_upscale, y_upscale;

  //   int pool_width, pool_height;
  //   int pool_vert_stride, pool_hor_stride;
  //   int pool_pad_h, pool_pad_w;

  //   bool pooling_enabled; //Pooling done if set to true
  //   bool pooling_params_initialized;
  //   bool input_data_on_gpu; // Set this to true if input data is already on GPU
  //   bool activation_set;
  //   bool grads_initialized;
  //   bool is_input_layer;

  //   float *d_data, *d_out;

  //   float *input_data, *h_output;
  //   float weight_init_mean, weight_init_stddev;
  //   float learning_rate, momentum, regularization_coeff;
  //   regularizer_type_Conv regularizer;

  //   float *d_activation_derivatives;
  //   float *d_pooling_derivatives;
  //   float *d_conv_filter_derivatives;
  //   float *d_conv_bias_derivatives;
  //   float *d_prev_layer_derivatives;

  //   float *d_filter_gradients, *d_filter_gradients_prev;
  //   float *d_filter_gradients_final;
  //   float *d_bias_gradients, *d_bias_gradients_prev;
  //   float *d_bias_gradients_final;

  //   float *d_conv, *d_filt, *d_bias, *d_pool; //d_conv has the output on GPU

  //   cudnnHandle_t cudnn_handle;
  //   cublasHandle_t cublas_handle;

  //   cudaError_t cudaError_stat;
  //   curandStatus_t curandStatus_stat;
  //   cudnnStatus_t cudnnStatus_stat;
  //   cublasStatus_t cublasStatus_stat;

  //   ~ConvLayer();

  // private:
  //   void AllocateGPUMemory(void);
  //   void InitializeFilters(float mean, float stddev); //Temporary
  //   float GetRandomNum();
  //   void InitializeBiases(void); //Temporary
  //   void InitBackpropVars(void);
  //   void CustomWeightInitializer(float *d_wt_mat, int wt_mat_sz);
  //   void Convolve_worker(void);

  //   float alpha, beta;

  //   float *grad_swap_tmp;
  //   float reg_inp_scalar;
  //   float neg_one_scalar;

  //   size_t fwd_workspace_size;
  //   size_t bwd_filter_workspace_size, bwd_data_workspace_size;
  //   void *d_fwd_workspace;
  //   void *d_bwd_filter_workspace, *d_bwd_data_workspace;
  //   bool output_copied_to_host;
  //   bool d_out_allocated;

  //   int filter_linear_size, filter_total_size;

  //   float *d_fwd_derivatives_tmp;

  //   //convTensor is descriptor to conv output
  //   cudnnTensorDescriptor_t dataTensor, convTensor, biasTensor, poolTensor;
  //   cudnnConvolutionDescriptor_t convDesc;
  //   cudnnFilterDescriptor_t filterDesc;
  //   cudnnConvolutionFwdAlgo_t fwd_algo;
  //   cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  //   cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  //   cudnnPoolingDescriptor_t poolDesc;
  //   cudnnPoolingMode_t pool_mode;
  //   cudnnActivationDescriptor_t cudnn_activation_desc;
  // };

  // // //All vectors assumed to be column vectors
  // enum regularizer_type_FC { L1, L2 };

  
  // void ReAlignMemory_ShiftLeft(float *d_mat, float *d_helper,
  //                            int rows, int cols,
  //                            int max_threadblock_size);

  // void ReAlignMemory_ShiftRight(float *d_mat, float *d_helper,
  //                             int rows, int cols, int max_threadblock_size);

  // void ReluBackprop(float *d_backprop_derivatives, float *d_out_xw_act,
  //                 float *d_fwd_layer_derivatives, float relu_clip,
  //                 int derivative_matrix_size);

  // void SigmoidBackprop(float *d_backprop_derivatives, float *d_out_xw_act,
  //                    float *d_fwd_layer_derivatives,
  //                    int derivative_matrix_size);

  // void SubtractElemwise(float *d_mat, float delta, int mat_size);

  // // Computes W - lambda * W^2
  // // Computes (1 - lr * reg_coeff / batch_size) * W
  // void WeightMatrixRegularizeElemWise(float *d_mat_in, int d_mat_cols,
  //                                     float reg_inp_scalar, int d_mat_size);

  // void FloatGPUMemset(float *d_array_arg,
  //                   int array_size_arg, float val);

  // //MOMENTUM IMPLEMENTATION FAULTY; momentum multiplied with
  // //prev_gradient*learningrate
  // //Fully Connected Layer
  // class FCLayer {
  // public:
  //   FCLayer(const cudnnHandle_t &cudnn_handle_arg,
  //           const cublasHandle_t &cublas_handle_arg,
  //           const cudaDeviceProp &cuda_device_prop_arg,
  //           int input_batch_size_arg, int input_n_arg,
  //           int output_n_arg, bool is_softmax_layer_arg = false,
  //           float learning_rate_arg = 1e-2f,
  //           float momentum_arg = 1e-3f,
  //           float regularization_coeff_arg = 1e-3f,
  //           regularizer_type_FC regularizer_arg = L2,
  //           float weight_init_mean_arg = 0.0f,
  //           float weight_init_stddev_arg = 1.0f);
  //   void LoadData(float *input_data_arg, bool input_data_on_gpu_arg);

  //   // This automatically applies softmax if is_softmax_layer is true
  //   void ForwardProp();

  //   // Do NOT set activation if softmax layer
  //   void SetActivationFunc(cudnnActivationMode_t activation_mode_arg,
  //                          double relu_clip = 0.0f);
  //   void SoftmaxOut();

  //   // Returns matrix with dimension (input_neurons + 1, output_neurons)
  //   void ComputeSoftmaxGradients(float *h_labels);

  //   void ComputeLayerGradients(float *d_backprop_derivatives);

  //   //Input gradient matrix of dimensions (input_neurons + 1) * output_neurons
  //   void UpdateWeights(float *d_update_gradients);

  //   void InitBackpropVars();

  //   regularizer_type_FC regularizer;

  //   int threadBlockSize;
  //   int max_img_side_dim_3_channel;
  //   int output_neurons, input_neurons; //output_neurons->#Neurons,
  //                                      //input_neurons->#Neurons in previous layer
  //   int input_batch_size, input_data_matrix_size;
  //   int weight_matrix_size, weight_matrix_rows, weight_matrix_cols;
  //   float *input_data, *h_out;
  //   float weight_init_mean, weight_init_stddev;
  //   float learning_rate, momentum, regularization_coeff;
  //   bool input_data_on_gpu, activation_set;
  //   bool grads_initialized;
  //   bool is_softmax_layer;
  //   bool is_input_layer;

  //   float *d_data, *d_out, *d_labels, *d_out_xw, *d_out_xw_act;
  //   float *d_out_pre_softmax;
  //   float *d_softmax_weight_matrix;
  //   float *d_weight_matrix; //#Rows = input_neurons + 1, #Cols = output_neurons
  //   float *d_gradients_T, *d_gradients;

  //   //REMOVE 1st column & change dim to input_batch_size * input_neurons
  //   float *d_prev_layer_derivatives; //input_batch_size * (input_neurons + 1); These are
  //                                    //propagated backwards
  //   float *d_fwd_layer_derivatives; //input_batch_size * output_neurons
  //   float *d_softmax_derivatives;

  //   float relu_activation_clip;

  //   cublasHandle_t cublas_handle;
  //   cudnnHandle_t cudnn_handle;
  //   cudnnActivationDescriptor_t cudnn_activation_desc;
  //   cudnnActivationMode_t activation_mode;
  //   cudaDeviceProp cuda_device_prop;


  // private:
  //   void AllocateGPUMemory();
  //   void InitializeWeightMatrix(float mean, float stddev);
  //   void AddOneVector_CPU(float *d_mat, int rows, int cols); //Adds a column of 1's to input data matrix
  //   void AddOneVector_GPU(float *d_mat, int rows, int cols); //Adds a column of 1's to input data matrix
  //   void CustomWeightInitializer(float *d_wt_mat, int len, float val);
  //   float GetRandomNum();
  //   void SumColumns(float *d_mat, float *d_out, int rows, int cols);


  //   // Input -> (input_batch_size * output_neurons)
  //   void ComputePrevLayerDerivatives(float *d_fwd_derivatives);

  //   void reinit_vars(); //REMOVE THIS LATER; used to set specific vars to check for 
  //                       //correct results by hand

  //                       // Returns the transpose of (scale_coeff*A.T*B + prior_coeff*C); 
  //                       // Python equivalent: C = (scale_coeff*np.dot(A.T, B) + prior_coeff*C).T
  //   cublasStatus_t MatMulT(float *d_A, float *d_B, float *d_C,
  //                          int rows_A, int cols_A,
  //                          int rows_B, int cols_B, float scale_coeff = 1.0f,
  //                          float prior_coeff = 0.0f);

  //   float alpha, beta;
  //   float *d_shift_helper;
  //   //float *d_softmax_gradients;
  //   //float *d_softmax_gradients_T; //Contains weight grads for softmax backprop
  //   float *d_out_minus_labels; //contains d_out - d_labels elem-wise
  //   float *d_elem_grads; //contains weight gradients for each example in batch
  //                        //It's columns are summed to get gradients
  //                        //float *d_weight_matrix_squared;
  //   float neg_one_scalar; // contains -1
  //   float neg_learning_rate;
  //   float *d_onevect;
  //   float reg_inp_scalar;
  //   float softmax_grad_coeff, neg_softmax_grad_coeff;

  //   bool d_data_allocated;

  //   cudnnTensorDescriptor_t d_out_tensor;

  //   cudaError_t cudaError_stat;
  //   curandStatus_t curandStatus_stat;
  //   cudnnStatus_t cudnnStatus_stat;
  //   cublasStatus_t cublasStatus_stat;
  // };
}