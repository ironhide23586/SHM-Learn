#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>

#include <vector>
#include <iostream>

#define GPU_WARP_DISPATCHERS 2
#define GPU_WARP_SIZE 32

#define __cudaSafeCall __cudaSafeCall_SHMat
#define __cudnnSafeCall __cudnnSafeCall_SHMat
#define __cublasSafeCall __cublasSafeCall_SHMat
#define __cudaCheckError __cudaCheckError_SHMat
#define cublasGetErrorString cublasGetErrorString_SHMat

#include "err_check.h"

using namespace std;

void FloatCUDAMemset(float *d_array, int array_size, float val);

enum mem_location { CPU, GPU };

class SHMatrix {

public:
  SHMatrix(float *mat_data, std::vector<int> &dims, mem_location = GPU);
  SHMatrix(std::vector<int> &dims, mem_location = GPU,
           bool default_init = false, float init_val = 0.0f);
  void Print(bool print_elem = true);

  float *data;
  float mean, mini, maxi;
  int mini_idx, maxi_idx;
  mem_location data_loc;
  std::vector<int> data_dims;
  int rows, cols, num_elems;
  
  ~SHMatrix();

private:
  void load_dims(std::vector<int> &dims);
  void print_h_var(float *h_v, int r, int c, bool print_elem);
};

