#include <stdlib.h>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>

#include <cudnn.h>

#include "ConvLayer.h"
#include "FCLayer.h"


#include <limits>
#include <random>
#include <chrono>
#include <cmath>
//#include <math.h>

#include <fstream>
#include <string>
#include <cstring>

inline std::string separator() {
#ifdef _WIN32
  return "\\";
#else
  return "/";
#endif
}

#define DATA_SIDE 32 //Throws GPU setup error if above 257
#define CHANNELS 3

#define BATCH_SIZE 128
#define LABELS 10

#define EPOCHS 10

#define EPOCH_COMPONENT_SIZE 10000
#define EPOCH_SIZE 50000

using namespace std;
//using namespace cv;

int read_imgs_local, read_imgs_global;
int data_file_idx;

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

float matrix_square_sum_exclude_bias(float *d_mat, int sz, int cols) {
  float *tmp = (float *)malloc(sizeof(float) * sz);
  cudaMemcpy(tmp, d_mat, sizeof(float) * sz, cudaMemcpyDeviceToHost);
  float ans = 0.0;
  for (int i = cols; i < sz; i++) {
    ans += (tmp[i] * tmp[i]);
  }
  free(tmp);
  return ans;
}

float matrix_square_sum(float *d_mat, int sz) {
  float *tmp = (float *)malloc(sizeof(float) * sz);
  cudaMemcpy(tmp, d_mat, sizeof(float) * sz, cudaMemcpyDeviceToHost);
  float ans = 0.0;
  for (int i = 0; i < sz; i++) {
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

void readBatch_cifar10_lim_v2(FILE *fp, float *h_imgs, float *h_lbls) {
  int row_size = (CHANNELS * DATA_SIDE * DATA_SIDE) + 1;
  int row_size_x = row_size - 1;
  int batch_bytes = BATCH_SIZE * row_size;
  int start_idx;
  unsigned char *buff = (unsigned char *)malloc(sizeof(unsigned char)
                                                * batch_bytes);
  int read_examples = 0, curr_example = 0;
  int lbl;
  memset(h_lbls, 0, sizeof(float) * BATCH_SIZE * LABELS);

  if ((read_imgs_local + BATCH_SIZE) > EPOCH_COMPONENT_SIZE) {
    fclose(fp);
    data_file_idx = (data_file_idx % 5) + 1;
    if (data_file_idx == 1) {
      read_imgs_global = 0;
    }
    std::string train_file = "cifar-10-binary" + separator() + "cifar-10-batches-bin" + separator()
      + "data_batch_"
      + std::to_string(data_file_idx) + ".bin";
    fp = fopen(train_file.c_str(), "r");

    read_imgs_local = 0;
    curr_example = 0;
  }

  fread(buff, sizeof(unsigned char), batch_bytes, fp);

  memset(h_lbls, 0, sizeof(float) * BATCH_SIZE * LABELS);
  for (int i = 0; i < BATCH_SIZE; i++) {
    start_idx = i * row_size;
    lbl = (int)buff[start_idx];
    h_lbls[lbl + i * LABELS] = 1.0f;
    int col = 0;
    for (int j = start_idx + 1; j < start_idx + row_size; j++) {
      h_imgs[col + i * row_size_x] = (float)buff[j];
      h_imgs[col + i * row_size_x] /= 255.0f;
      col++;
    }
  }
  free(buff);
  read_imgs_local += BATCH_SIZE;
  read_imgs_global += BATCH_SIZE;
  fseek(fp, read_imgs_local * row_size, 0);
}

void move_to_gpu_stage(float *x, float *y, float *gpu_stage, int x_len, int y_len) {
  memcpy(gpu_stage, x, sizeof(float) * x_len);
  memcpy(&gpu_stage[x_len], y, sizeof(float) * y_len);
}

int my_floorf_division(float a, float b) {
  return ((a - 1) / b);
}

//void show_img(cv::Mat &img) {
//  cv::Mat img_scaled = cv::Mat(600, 600, CV_8UC3);
//  cv::resize(img, img_scaled, img_scaled.size());
//  cv::namedWindow("image");
//  cv::imshow("image", img_scaled);
//  cv::waitKey();
//}

//cv::Mat lin2mat(float *img_lin) {
//  cv::Mat ans = cv::Mat(DATA_SIDE, DATA_SIDE, CV_8UC3); 
//  for (int r = 0; r < DATA_SIDE; r++) {
//    for (int c = 0; c < DATA_SIDE; c++) {
//      for (int chan = 0; chan < CHANNELS; chan++) {
//        ans.data[chan + c * CHANNELS + r * DATA_SIDE * CHANNELS] 
//          = img_lin[c + r * DATA_SIDE + chan * DATA_SIDE * DATA_SIDE];
//      }
//    }
//  }
//  return ans;
//}
//
//const char* get_train_file_path(int data_file_idx) {
//  std::string train_file = "cifar-10-binary" + separator() + "cifar-10-batches-bin" + separator()
//    + "data_batch_"
//    + std::to_string(data_file_idx) + ".bin";
//  return train_file.c_str();
//}

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

  //cv::Mat img = cv::imread("t0.jpg");
  //show_img(img);


  int batch_size = BATCH_SIZE;
  float my_loss, loss, wt_sum, cl0_wt_sum, cl1_wt_sum, cl2_wt_sum, fcl0_wt_sum, fcl1_wt_sum,
    fcl2_wt_sum, dur, avg_dur;
  cublasHandle_t cublasHandle;
  cublasStatus_stat = cublasCreate_v2(&cublasHandle);
  std::cout << "cublas handle create -->" << cublasStatus_stat << std::endl;

  cudnnHandle_t cudnnHandle;
  cudnnStatus_t cudnn_status;
  cudnn_status = cudnnCreate(&cudnnHandle);
  std::cout << "cuDNN initialization -->" << cudnn_status << std::endl;

  float *x, *y;
  float *h_out;

  cudaMallocHost((void **)&x, sizeof(float) * BATCH_SIZE * CHANNELS 
                 * DATA_SIDE * DATA_SIDE);
  cudaMallocHost((void **)&y, sizeof(float) * BATCH_SIZE * LABELS);

  float *x_test = (float *)malloc(sizeof(float) * BATCH_SIZE * CHANNELS 
                                  * DATA_SIDE * DATA_SIDE);
  float *y_test = (float *)malloc(sizeof(float) * BATCH_SIZE * LABELS);

  data_file_idx = 1;
  std::string train_file = "cifar-10-binary" + separator() + "cifar-10-batches-bin" + separator()
    + "data_batch_"
    + std::to_string(data_file_idx) + ".bin";
  FILE *fp_data_train = fopen(train_file.c_str(), "r");

  std::string test_file = "cifar-10-binary" + separator() + "cifar-10-batches-bin" + separator()
    + "test_batch.bin";
  FILE *fp_data_test = fopen(test_file.c_str(), "r");  

  ofstream results_file;
  /*results_file.open("shmlearn_results.txt");*/

  std::vector<std::string> labels =
  {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "doggo",
    "frog",
    "horse",
    "ship",
    "truck"
  };


  //while (true) {
  //  readBatch_cifar10_lim_v2(fp_data_train, x, y);
  //  for (int i = 0; i < BATCH_SIZE; i++) {
  //    for (int k = 0; k < LABELS; k++) {
  //      if (y[k + i * LABELS] > 0)
  //        std::cout << labels[k] << std::endl;
  //    }
  //    show_img(lin2mat(&x[i * 3072]));
  //  }
  //  std::cout << "----------" << std::endl;
  //}
  
  
  read_imgs_local = 0;
  read_imgs_global = 0;

  float base_lr = 0.001f, gamma = 0.4f, power = 0;
  float lr = base_lr * powf(1 + gamma, -power);
  float reg = 0.004f;
  float mom = 0.9f;


  ConvLayer cl0(cudnnHandle, cublasHandle, BATCH_SIZE, CHANNELS, DATA_SIDE, DATA_SIDE,
                2, 2, 1, 1, 5, 5, 32, lr, mom, reg);
  cl0.SetPoolingParams(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 3, 3, 2, 2, 0, 0);
  cl0.SetActivationFunc(CUDNN_ACTIVATION_RELU);
  cl0.is_input_layer = true;

  ConvLayer cl1(cudnnHandle, cublasHandle, cl0.output_n, cl0.output_c,
                cl0.output_h, cl0.output_w, 2, 2, 1, 1, 5, 5, 32, lr, mom, reg);
  cl1.SetPoolingParams(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 3, 3, 2, 2, 0, 0);
  cl1.SetActivationFunc(CUDNN_ACTIVATION_RELU);

  ConvLayer cl2(cudnnHandle, cublasHandle, cl1.output_n, cl1.output_c,
                cl1.output_h, cl1.output_w, 2, 2, 1, 1, 5, 5, 64, lr, mom, reg);
  cl2.SetPoolingParams(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 3, 3, 2, 2, 0, 0);
  cl2.SetActivationFunc(CUDNN_ACTIVATION_RELU);

  FCLayer fcl0(cudnnHandle, cublasHandle, cudaProp, cl0.output_n,
               cl0.output_c * cl0.output_h * cl0.output_w,
               64, false, lr, mom, reg);
  fcl0.SetActivationFunc(CUDNN_ACTIVATION_RELU);

  FCLayer fcl1(cudnnHandle, cublasHandle, cudaProp, fcl0.input_batch_size,
               fcl0.output_neurons, 32, false, lr, mom, reg);
  fcl1.SetActivationFunc(CUDNN_ACTIVATION_RELU);

  FCLayer fcl2(cudnnHandle, cublasHandle, cudaProp, fcl1.input_batch_size,
               fcl1.output_neurons, LABELS, true, lr, mom, reg);

  auto st = std::chrono::system_clock::now();

  //float *h_out = (float *)malloc(sizeof(float) * BATCH_SIZE * LABELS);
  cudaMallocHost((void **)&h_out, sizeof(float) * BATCH_SIZE * LABELS);

  int batch = 1;
  int lim = my_floorf_division(EPOCH_SIZE, BATCH_SIZE);
  int epoch = 1, prog = 1;
  int prev_read_imgs;

  float batch_verify;
  float ts = 0.0f;
  int cnt = 1;

  auto train_start = std::chrono::high_resolution_clock::now();
  auto train_end = std::chrono::high_resolution_clock::now();

  auto t0 = std::chrono::high_resolution_clock::now();
  auto t1 = std::chrono::high_resolution_clock::now();

  float dur0, dur1_sft, dur1_lyr, dur2;

  while (epoch <= EPOCHS) {
    t0 = std::chrono::high_resolution_clock::now();
    readBatch_cifar10_lim_v2(fp_data_train, x, y);
    t1 = std::chrono::high_resolution_clock::now();
    dur0 = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t1
                                                                      - t0)
      .count() * 1e-9f;
    if (prog == 1) {
      prev_read_imgs = read_imgs_global;
    }

    cl0.learning_rate = lr;
    cl1.learning_rate = lr;
    cl2.learning_rate = lr;
    fcl0.learning_rate = lr;
    fcl1.learning_rate = lr;
    fcl2.learning_rate = lr;

    // Forward Pass
    train_start = std::chrono::high_resolution_clock::now();
    cl0.LoadData(x, false);
    cl0.Convolve();

    //print_d_var3(cl0.d_out, BATCH_SIZE, cl0.output_c * cl0.output_h * cl0.output_w);

    cl1.LoadData(cl0.d_out, true);
    cl1.Convolve();

    //print_d_var3(cl1.d_out, BATCH_SIZE, cl1.output_c * cl1.output_h * cl1.output_w);

    cl2.LoadData(cl1.d_out, true);
    cl2.Convolve();

    //print_d_var3(cl2.d_out, BATCH_SIZE, cl2.output_c * cl2.output_h * cl2.output_w);
    
    fcl0.LoadData(cl2.d_out, true);
    fcl0.ForwardProp();

    //print_d_var3(fcl0.d_out, BATCH_SIZE, fcl0.output_neurons);

    fcl1.LoadData(fcl0.d_out, true);
    fcl1.ForwardProp();

    //print_d_var3(fcl1.d_out, BATCH_SIZE, fcl1.output_neurons);
    
    fcl2.LoadData(fcl1.d_out, true);
    fcl2.ForwardProp();

    //print_d_var3(fcl2.d_out, BATCH_SIZE, fcl2.output_neurons);

    // Back-propagation
    fcl2.ComputeSoftmaxGradients(y);
    //print_d_var3(fcl2.d_gradients, fcl2.input_neurons + 1, fcl2.output_neurons);
    fcl1.ComputeLayerGradients(fcl2.d_prev_layer_derivatives);
    fcl0.ComputeLayerGradients(fcl1.d_prev_layer_derivatives);
    cl2.ComputeLayerGradients(fcl0.d_prev_layer_derivatives);
    cl1.ComputeLayerGradients(cl2.d_prev_layer_derivatives);
    cl0.ComputeLayerGradients(cl1.d_prev_layer_derivatives);

    

    fcl2.UpdateWeights(fcl2.d_gradients);
    fcl1.UpdateWeights(fcl1.d_gradients);
    fcl0.UpdateWeights(fcl0.d_gradients);
    cl2.UpdateWeights(cl2.d_filter_gradients, cl2.d_bias_gradients);
    cl1.UpdateWeights(cl1.d_filter_gradients, cl1.d_bias_gradients);
    cl0.UpdateWeights(cl0.d_filter_gradients, cl0.d_bias_gradients);

    //print_d_var3(cl0.d_out, BATCH_SIZE, cl0.output_c * cl0.output_h * cl0.output_w);
    //print_d_var3(cl0.d_filt, cl0.feature_maps, cl0.input_c * cl0.kernel_h * cl0.kernel_w);
    //print_d_var3(cl0.d_filter_gradients, cl0.feature_maps, cl0.input_c * cl0.kernel_h
                 //* cl0.kernel_w);

    train_end = std::chrono::high_resolution_clock::now();

    //print_d_var3(fcl2.d_weight_matrix, fcl2.weight_matrix_rows, fcl2.weight_matrix_cols);
    

    
    //cl0.Convolve();
    //fcl0.LoadData(cl0.d_out, true);
    //fcl0.ForwardProp();
    //fcl2.LoadData(fcl0.d_out, true);
    //fcl2.ForwardProp();
    
    t0 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_out, fcl2.d_out, sizeof(float) * BATCH_SIZE * LABELS, //CAUSING DELAY
               cudaMemcpyDeviceToHost);
    t1 = std::chrono::high_resolution_clock::now();
    dur2 = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t1
                                                                       - t0)
      .count() * 1e-9f;
    my_loss = 0.0f;
    
    for (int i = 0; i < BATCH_SIZE; i++) {
      for (int j = 0; j < LABELS; j++) {
        my_loss -= ((y[j + i * LABELS] * log(h_out[j + i * LABELS])));
      }
    }
    my_loss /= BATCH_SIZE;
    wt_sum = 0.0f;
    
    cl0_wt_sum = matrix_square_sum(cl0.d_filt, cl0.input_c * cl0.feature_maps
                                   * cl0.kernel_h * cl0.kernel_w);
    cl1_wt_sum = matrix_square_sum(cl1.d_filt, cl1.input_c * cl1.feature_maps
                                   * cl1.kernel_h * cl1.kernel_w);
    cl2_wt_sum = matrix_square_sum(cl2.d_filt, cl2.input_c * cl2.feature_maps
                                   * cl2.kernel_h * cl2.kernel_w);

    fcl0_wt_sum = matrix_square_sum_exclude_bias(fcl0.d_weight_matrix, fcl0.weight_matrix_size,
                                                 fcl0.weight_matrix_cols);
    fcl1_wt_sum = matrix_square_sum_exclude_bias(fcl1.d_weight_matrix, fcl1.weight_matrix_size,
                                                 fcl1.weight_matrix_cols);
    fcl2_wt_sum = matrix_square_sum_exclude_bias(fcl2.d_weight_matrix, fcl2.weight_matrix_size,
                                                 fcl2.weight_matrix_cols);
    
    wt_sum = cl0_wt_sum + cl1_wt_sum + cl2_wt_sum + fcl0_wt_sum + fcl1_wt_sum + fcl2_wt_sum;
    float wt_loss = (reg * 0.5f) * wt_sum;
    
    loss = my_loss + wt_loss;
    
      
    dur = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(train_end 
                                                                      - train_start)
      .count() * 1e-9f;
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
      << " Epoch = " << epoch << " Loss = " << loss 
      << " C++_CUDA_GPU Avg iter time = " << avg_dur;

    batch++;
    prog++;

    if (read_imgs_global < prev_read_imgs) {
      batch = 1;
      epoch++;
    }
    prev_read_imgs = read_imgs_global;
    results_file.open("shmlearn_results.txt", std::ofstream::out | std::ofstream::app);
    results_file << ts << " " << loss << "\n";
    results_file.close();
    std::cout << std::endl;
    cnt++;
    
  }
  results_file.open("shmlearn_results.txt", std::ofstream::out | std::ofstream::app);
  results_file << "Avg iter time = " << avg_dur << " seconds\n";
  results_file.close();

  auto et = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(et - st);

  cudnnDestroy(cudnnHandle);
  cublasDestroy_v2(cublasHandle);

  int k = getchar();
  return 0;
}