#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>

#include <vector>
#include <iostream>
#include <random>
#include <string>

#define GPU_WARP_DISPATCHERS 2
#define GPU_WARP_SIZE 32

#include "err_check.h"

using namespace std;

void FloatCUDAMemset(float *d_array, int array_size, float val);
void ScaleUniformSHMatrix(float *d_array, int array_size,
                          float lower, float higher);
void ElemwiseMultiplyInPlaceGPU(float *d_src, float *d_arg,
                                int array_size);
void ElemwiseMultiplyInPlaceCPU(float *d_src, float *d_arg,
                                int array_size);

enum mem_location { CPU, GPU };

class SHMatrix {

public:
  SHMatrix(const cublasHandle_t &cublas_handle,
           float *mat_data, std::vector<int> &dims,
           mem_location = GPU);
  SHMatrix(const cublasHandle_t &cublas_handle_arg,
           std::vector<int> &dims, mem_location = GPU,
           bool default_init = false, float init_val = 0.0f);
  void Print(bool print_elem = true);

  void GaussianInit(float mean = 0.0f, float stddev = 0.1f);
  void UniformInit(float lower = -0.5f, float higher = 0.5f);

  static float GetGaussianNum(float mean, float stddev);
  static float GetUniformNum(float lower, float higher);

  void operator*=(SHMatrix &arg);
  void operator+=(SHMatrix &arg);
  void operator-=(SHMatrix &arg);
  void operator/=(SHMatrix &arg);

  SHMatrix operator*(SHMatrix &arg);
  SHMatrix operator+(SHMatrix &arg);
  SHMatrix operator-(SHMatrix &arg);
  SHMatrix operator/(SHMatrix &arg);

  float *data;
  float mean, mini, maxi;
  int mini_idx, maxi_idx;
  mem_location data_loc;
  std::vector<int> data_dims;
  int rows, cols, num_elems;
  std::string name;
  
  ~SHMatrix();

private:
  void load_dims(std::vector<int> &dims);
  void print_h_var(float *h_v, int r, int c, bool print_elem);
  void gaussian_init_gpu(float mean = 0.0f, float stddev = 0.1f);
  void gaussian_init_cpu(float mean = 0.0f, float stddev = 0.1f);
  void uniform_init_gpu(float lower = -0.5f, float higher = 0.5f);
  void uniform_init_cpu(float lower = -0.5f, float higher = 0.5f);

  void gpu2any_elemwise_mult(SHMatrix &arg);
  void cpu2any_elemwise_mult(SHMatrix &arg);
};