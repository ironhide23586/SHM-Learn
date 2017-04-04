#include <iostream>
#include <cudnn.h>

using namespace std;

int main()
{

  cudnnActivationForward(cudnn_handle, cudnn_activation_desc, &alpha,
                          d_out_tensor, d_out_xw, &beta, d_out_tensor,
                          d_out_xw_act);
}

void FCLayer::reinit_vars() {
  // cudaError_stat = cudaFree(d_softmax_gradients_T);
  //cudaError_stat = cudaFree(d_data);
  cudaError_stat = cudaFree(d_shift_helper);
  //cudaError_stat = cudaFree(d_out);
  //cudaError_stat = cudaFree(d_labels);
  //cudaError_stat = cudaFree(d_fwd_layer_derivatives);
  //free(input_data);
  //cudaError_stat = cudaFree(d_gradients_T);
  //cudaError_stat = cudaFree(d_gradients);
  cudaError_stat = cudaFree(d_weight_matrix);
  cudaError_stat = cudaFree(d_prev_layer_derivatives);
  //cudaError_stat = cudaFree(d_weight_matrix_squared);

  input_neurons = 5;
  output_neurons = 3;
  input_batch_size = 4;
  softmax_grad_coeff = 1.0f / input_batch_size;
  learning_rate = 1.0f;
  momentum = 0.0f;
  regularization_coeff = 0.0f;
  input_data_matrix_size = input_batch_size * (input_neurons + 1);

  weight_matrix_rows = input_neurons + 1;
  weight_matrix_cols = output_neurons;

  //cudaError_stat = cudaMalloc((void **)&d_gradients,
  //                            sizeof(float) * (input_neurons + 1)
  //                            * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_gradients_T,
  //                            sizeof(float) * (input_neurons + 1)
  //                            * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_softmax_gradients_T,
  //                            sizeof(float) * output_neurons
  //                            * (input_neurons + 1));
  cudaError_stat = cudaMalloc((void **)&d_prev_layer_derivatives, sizeof(float)
                              * input_batch_size
                              * (input_neurons + 1));
  cudaError_stat = cudaMalloc((void **)&d_shift_helper,
                              sizeof(float) * input_batch_size
                              * (input_batch_size - 1) / 2);
  //cudaError_stat = cudaMalloc((void **)&d_labels,
  //                            sizeof(float) * input_batch_size
  //                            * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_out,
  //                            sizeof(float) * input_batch_size
  //                            * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_gradients_I,
  //                            sizeof(float) * output_neurons
  //                            * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_data,
  //                            sizeof(float) * input_data_matrix_size);
  cudaError_stat = cudaMalloc((void **)&d_weight_matrix,
                              sizeof(float) * (input_neurons + 1)
                              * output_neurons);
  //cudaError_stat = cudaMalloc((void **)&d_weight_matrix_squared,
  //                            sizeof(float) * (input_neurons + 1)
  //                            * output_neurons);

  float h_d[20] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
  float h_o[6] = { 0.1f, 0.5f, 0.4f, 0.7f, 0.2f, 0.1f };
  float h_l[6] = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f };

  float h_prev_derivs[24] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };

  //float h_w[12] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  float h_w[18];
  for (int i = 0; i < 18; i++) {
    h_w[i] = (float)i / 10;
  }
  float h_prev_grad[12] = { -0.7,  -0.8, -0.9, -1,
    0.75, 1.1, 1.45, 1.8,
    -0.05, -0.3, -0.55, -0.8 };
  //cudaMemcpy(d_data, h_d, sizeof(float) * 20, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_out, h_o, sizeof(float) * 6, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_labels, h_l, sizeof(float) * 6, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_softmax_gradients_T, h_prev_grad, sizeof(float) * 12, cudaMemcpyHostToDevice);
  //InitIdentityMatrix(d_gradients_I, output_neurons);
  cudaMemcpy(d_weight_matrix, h_w, sizeof(float) * (input_neurons + 1) * output_neurons, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_fwd_layer_derivatives, h_o, sizeof(float) * 6, cudaMemcpyHostToDevice);
  cudaMemcpy(d_prev_layer_derivatives, h_prev_derivs, sizeof(float) * input_batch_size * (input_neurons + 1), cudaMemcpyHostToDevice);
  //memcpy(input_data, h_d, sizeof(float) * (input_data_matrix_size - input_batch_size));
}