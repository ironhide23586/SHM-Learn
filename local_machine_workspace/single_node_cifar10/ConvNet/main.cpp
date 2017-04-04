#include <stdlib.h>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "sample.h"
//#include "GlobalInclude.h"

#include <cudnn.h>

#include "ConvLayer.h"
#include "FCLayer.h"


#include <limits>
#include <random>
#include <chrono>
#include <cmath>
//#include <math.h>

#include <fstream>
#include <string.h>

//#define DATA_SIDE 32 //Throws GPU setup error if above 257
//#define CHANNELS 3

#define DATA_SIDE 28 //Throws GPU setup error if above 257
#define CHANNELS 1

#define BATCH_SIZE 64
#define LABELS 10

#define EPOCHS 10
#define EPOCH_SIZE 60000

using namespace std;

int read_imgs;

void print_d_var3(float *d_v, int r, int c, bool print_elem = true) {
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

void print_h_var3(float *h_v, int r, int c) {
  std::cout << "\n-------------------------" << std::endl;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      std::cout << h_v[j + i * c] << "\t";
      //printf("%f\t", h_v_local[j + i * c]);
    }
    std::cout << std::endl;
  }
}

void sumCols(float *mat, int rows, int cols, float *sums) {
  //#pragma omp parallel
  for (int j = 0; j < cols; j++) {
    sums[j] = 0.0f;
    for (int i = 0; i < rows; i++) {
      sums[j] += mat[j + i * cols];
    }
  }
}

float matrix_square_sum(float *d_mat, int sz, int cols) {
  float *tmp = (float *)malloc(sizeof(float) * sz);
  cudaMemcpy(tmp, d_mat, sizeof(float) * sz, cudaMemcpyDeviceToHost);
  float ans = 0.0;
  for (int i = cols; i < sz; i++) {
    ans += (tmp[i] * tmp[i]);
  }
  free(tmp);
  return ans;
}

float my_rand() {
  static std::default_random_engine re;
  static std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  return dist(re);
}

void readBatch_mnist(FILE *fp_x, FILE *fp_y, float *h_imgs, float *h_lbls) {
  int row_size_x = (CHANNELS * DATA_SIDE * DATA_SIDE);
  int batch_bytes_x = BATCH_SIZE * row_size_x;
  int start_idx;
  unsigned char *buff_x = (unsigned char *)malloc(sizeof(unsigned char) * BATCH_SIZE
                                                  * row_size_x);
  unsigned char *buff_y = (unsigned char *)malloc(sizeof(unsigned char) * BATCH_SIZE);
  fread(buff_x, sizeof(unsigned char), batch_bytes_x, fp_x);
  fread(buff_y, sizeof(unsigned char), BATCH_SIZE, fp_y);

  float mean_pixel = 0.0f;
  memset(h_lbls, 0, sizeof(float) * BATCH_SIZE * LABELS);
  memset(h_imgs, 0, sizeof(float) * batch_bytes_x);

  for (int i = 0; i < BATCH_SIZE; i++) {
    h_lbls[buff_y[i] + i * LABELS] = 1.0f;
    for (int j = 0; j < row_size_x; j++) {
      h_imgs[j + i * row_size_x]
        = (float)buff_x[j + i * row_size_x] / 255.0f;
      mean_pixel += h_imgs[j + i * row_size_x];
    }
  }
}

void readBatch_mnist_lim(FILE *fp_x, FILE *fp_y, float *h_imgs, float *h_lbls) {
  int row_size_x = (CHANNELS * DATA_SIDE * DATA_SIDE);
  int batch_bytes_x = BATCH_SIZE * row_size_x;
  int start_idx;
  unsigned char *buff_x = (unsigned char *)malloc(sizeof(unsigned char)
                                                  * row_size_x);
  unsigned char buff_y;

  float mean_pixel = 0.0f;

  memset(h_lbls, 0, sizeof(float) * BATCH_SIZE * LABELS);
  memset(h_imgs, 0, sizeof(float) * batch_bytes_x);

  int read_examples = 0, curr_example = 0;
  while (read_examples < BATCH_SIZE) {
    fread(buff_x, sizeof(unsigned char), row_size_x, fp_x);
    fread(&buff_y, sizeof(unsigned char), 1, fp_y);
    if (buff_y < LABELS) {
      h_lbls[buff_y + read_examples * LABELS] = 1.0f;
      for (int j = 0; j < row_size_x; j++) {
        h_imgs[j + read_examples * row_size_x]
          = (float)buff_x[j] / 255.0f;
      }
      read_examples++;
    }
    curr_example++;
    if ((read_imgs + curr_example) >= EPOCH_SIZE) {
      fseek(fp_x, 16, 0);
      fseek(fp_y, 8, 0);
      read_imgs = 0;
      curr_example = 0;
    }
  }
  free(buff_x);
  read_imgs += curr_example;
}

void readBatch_cifar10_lim(FILE *fp, float *h_imgs, float *h_lbls) {
  int row_size = (CHANNELS * DATA_SIDE * DATA_SIDE) + 1;
  int batch_bytes = BATCH_SIZE * row_size;
  int start_idx;
  unsigned char *buff = (unsigned char *)malloc(sizeof(unsigned char) * BATCH_SIZE
                                                * ((CHANNELS * DATA_SIDE * DATA_SIDE)
                                                   + 1));
  fread(buff, sizeof(unsigned char), batch_bytes, fp);
  memset(h_lbls, 0, sizeof(float) * BATCH_SIZE * LABELS);
  for (int i = 0; i < BATCH_SIZE; i++) {
    start_idx = i * row_size;
    h_lbls[((int)buff[start_idx] - 1) + i * LABELS] = 1.0f;
    int col = 0;
    for (int j = start_idx + 1; j < start_idx + row_size; j++) {
      h_imgs[col + i * (row_size - 1)] = (float)buff[j];
      h_imgs[col + i * (row_size - 1)] /= 255.0f;
      col++;
    }
  }
}

void move_to_gpu_stage(float *x, float *y, float *gpu_stage, int x_len, int y_len) {
  memcpy(gpu_stage, x, sizeof(float) * x_len);
  memcpy(&gpu_stage[x_len], y, sizeof(float) * y_len);
}

int my_floorf_division(float a, float b) {
  return ((a - 1) / b);
}

int main() {
  cudaError_t cudaError_stat;
  curandStatus_t curandStatus_stat;
  cudnnStatus_t cudnnStatus_stat;
  cublasStatus_t cublasStatus_stat;

  int numGPUs;

  cudaGetDeviceCount(&numGPUs);
  cudaSetDevice(0);
  cudaDeviceProp cudaProp;
  cudaGetDeviceProperties(&cudaProp, 0);
  std::cout << "Using GPU Device -> " << cudaProp.name << std::endl;
  cudaError_stat = cudaDeviceReset();
  std::cout << "cuda dev reset -->" << cudaError_stat << std::endl;

  int batch_size = BATCH_SIZE;
  float my_loss, loss, wt_sum, fcl0_wt_sum, fcl2_wt_sum, dur, avg_dur;
  cublasHandle_t cublasHandle;
  cublasStatus_stat = cublasCreate_v2(&cublasHandle);
  std::cout << "cublas handle create -->" << cublasStatus_stat << std::endl;

  cudnnHandle_t cudnnHandle;
  cudnnStatus_t cudnn_status;
  cudnn_status = cudnnCreate(&cudnnHandle);
  std::cout << "cuDNN initialization -->" << cudnn_status << std::endl;

  float *x, *y;

  cudaMallocHost((void **)&x, sizeof(float) * BATCH_SIZE * CHANNELS * DATA_SIDE * DATA_SIDE);
  cudaMallocHost((void **)&y, sizeof(float) * BATCH_SIZE * LABELS);

  float *x_test = (float *)malloc(sizeof(float) * BATCH_SIZE * CHANNELS * DATA_SIDE * DATA_SIDE);
  float *y_test = (float *)malloc(sizeof(float) * BATCH_SIZE * LABELS);

  FILE *fp_data_train = fopen("cifar-10-binary\cifar-10-batches-bin\data_batch_1.bin", "rb");
  FILE *fp_data_test = fopen("cifar-10-binary\cifar-10-batches-bin\test_batch.bin", "rb");

  readBatch_cifar10_lim(fp_data_test, x_test, y_test);

  ofstream results_file;
  results_file.open("shmlearn_results.txt");
  
  read_imgs = 0;
  

  float base_lr = 0.05f, gamma = 0.4f, power = 0;
  float lr = base_lr * powf(1 + gamma, -power);
  float reg = 0.01f;
  float mom = 0.0f;

  FCLayer fcl0(cudnnHandle, cublasHandle, cudaProp, BATCH_SIZE, CHANNELS * DATA_SIDE * DATA_SIDE,
               64, false, lr, mom, reg);
  fcl0.SetActivationFunc(CUDNN_ACTIVATION_SIGMOID);
  fcl0.is_input_layer = true;

  FCLayer fcl2(cudnnHandle, cublasHandle, cudaProp, fcl0.input_batch_size,
               fcl0.output_neurons, LABELS, true, lr, mom, reg);
  fcl2.InitBackpropVars();

  auto st = std::chrono::system_clock::now();

  float *h_out = (float *)malloc(sizeof(float) * BATCH_SIZE * LABELS);
  int batch = 1;
  int lim = my_floorf_division(EPOCH_SIZE, BATCH_SIZE);
  int epoch = 1, prog = 1;
  int prev_read_imgs;

  float batch_verify;
  float ts = 0.0f;
  int cnt = 1;

  auto train_start = std::chrono::high_resolution_clock::now();
  auto train_end = std::chrono::high_resolution_clock::now();
  float dur0, dur1_sft, dur1_lyr, dur2;

  while (epoch <= EPOCHS) {
    readBatch_cifar10_lim(fp_data_train, x, y);
    if (prog == 1) {
      prev_read_imgs = read_imgs;
    }

    fcl0.learning_rate = lr;
    fcl2.learning_rate = lr;

    // Forward Pass
    train_start = std::chrono::high_resolution_clock::now();
    fcl0.LoadData(x, false);
    fcl0.ForwardProp();
    fcl2.LoadData(fcl0.d_out, true);
    fcl2.ForwardProp();

    // Back-propagation
    fcl2.ComputeSoftmaxGradients(y);
    fcl0.ComputeLayerGradients(fcl2.d_prev_layer_derivatives);
    fcl2.UpdateWeights(fcl2.d_gradients);
    fcl0.UpdateWeights(fcl0.d_gradients);

    train_end = std::chrono::high_resolution_clock::now();

    fcl0.ForwardProp();
    fcl2.LoadData(fcl0.d_out, true);
    fcl2.ForwardProp();
    cudaMemcpy(h_out, fcl2.d_out, sizeof(float) * BATCH_SIZE * LABELS, cudaMemcpyDeviceToHost);

    my_loss = 0.0f;
    for (int i = 0; i < BATCH_SIZE; i++) {
      for (int j = 0; j < LABELS; j++) {
        my_loss -= ((y[j + i * LABELS] * log(h_out[j + i * LABELS])));
      }
    }
    my_loss /= BATCH_SIZE;
    wt_sum = 0.0f;
    fcl0_wt_sum = matrix_square_sum(fcl0.d_weight_matrix, fcl0.weight_matrix_size, fcl0.weight_matrix_cols);
    fcl2_wt_sum = matrix_square_sum(fcl2.d_weight_matrix, fcl2.weight_matrix_size, fcl2.weight_matrix_cols);
    wt_sum = fcl0_wt_sum + fcl2_wt_sum;
    float wt_loss = (reg * 0.5f) * wt_sum;
    loss = my_loss + wt_loss;

    dur = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(train_end - train_start).count() * 1e-9f;
    ts += dur;
    if (cnt == 253) {
      int k = 0;
    }
    if (cnt == 1)
      avg_dur = dur;
    else {
      avg_dur += dur;
      avg_dur /= 2;
    }
    std::cout << "Batch " << batch
      << " Epoch = " << epoch << " Loss = " << my_loss << " C++_CUDA_GPU Avg iter time = " << avg_dur;

    batch++;
    prog++;

    if (read_imgs < prev_read_imgs) {
      batch = 1;
      epoch++;
    }
    prev_read_imgs = read_imgs;

    results_file << ts << " " << my_loss << "\n";

    std::cout << std::endl;
    cnt++;
  }
  results_file << "Avg iter time = " << avg_dur << " seconds\n";
  results_file.close();

  auto et = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(et - st);

  cudnnDestroy(cudnnHandle);
  cublasDestroy_v2(cublasHandle);

  int k = getchar();
  return 0;
}