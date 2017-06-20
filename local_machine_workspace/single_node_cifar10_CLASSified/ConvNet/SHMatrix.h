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

enum mem_location { CPU, GPU };
enum ELEM_OP { MULT, DIV, ADD, SUB };

void FloatCUDAMemset(float *d_array, int array_size, float val);
void ScaleUniformSHMatrix(float *d_array, int array_size,
                          float lower, float higher);

void ElemwiseMultiplyInPlaceGPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_op = false,
                                bool arg_op = false);
void ElemwiseDivideInPlaceGPU(float *d_src, float *d_arg,
                              int ld_src, int ld_arg,
                              int array_size, bool src_op = false,
                              bool arg_op = false);
void ElemwiseAddInPlaceGPU(float *d_src, float *d_arg,
                           int ld_src, int ld_arg,
                           int array_size, bool src_op = false,
                           bool arg_op = false);
void ElemwiseSubtractInPlaceGPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_op = false,
                                bool arg_op = false);

void ElemwiseAddInPlaceGPU_Scalar(float *d_src, float scalar,
                                  int array_size);
void ElemwiseSubtractInPlaceGPU_Scalar(float *d_src, float scalar,
                                       int array_size);

void ElemwiseMultiplyInPlaceCPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_T_op = false,
                                bool arg_T_op = false);
void ElemwiseDivideInPlaceCPU(float *d_src, float *d_arg,
                              int ld_src, int ld_arg,
                              int array_size, bool src_T_op = false,
                              bool arg_T_op = false);
void ElemwiseAddInPlaceCPU(float *d_src, float *d_arg,
                           int ld_src, int ld_arg,
                           int array_size, bool src_T_op,
                           bool arg_T_op);
void ElemwiseSubtractInPlaceCPU(float *d_src, float *d_arg,
                                int ld_src, int ld_arg,
                                int array_size, bool src_T_op,
                                bool arg_T_op);

int get_tranposed_2DLin_idx(int src_idx, int ld_src, int array_size);

class SHMatrix {

public:
  SHMatrix(const cublasHandle_t &cublas_handle_arg,
           float *mat_data, std::vector<int> &dims,
           mem_location = GPU);
  SHMatrix(const cublasHandle_t &cublas_handle_arg,
           std::vector<int> &dims, mem_location = GPU,
           bool default_init = false, float init_val = 0.0f);
  SHMatrix(const cublasHandle_t &cublas_handle_arg,
           SHMatrix &src_shmatrix, mem_location = GPU);

  void Equate(SHMatrix &src_shmatrix);
  void Reallocate(std::vector<int> &dims, mem_location mem_loc = GPU,
                  bool copy_original = false, bool default_init = false,
                  float init_val = 0.0f);

  void Print(bool print_elem = true);
  void Move2GPU();
  void Move2CPU();

  void Clear();

  void GaussianInit(float mean = 0.0f, float stddev = 0.1f);
  void UniformInit(float lower = -0.5f, float higher = 0.5f);

  static float GetGaussianNum(float mean, float stddev);
  static float GetUniformNum(float lower, float higher);

  void CommitUnaryOps(); //commits the transpose & scaling operations
  void CommitTranspose(); // commits all lazy-tranpose ops
  void CommitScale(); // commits all lazy-scale ops

  SHMatrix& T(); //Transpose operation
  SHMatrix& Scale(float scale_arg); //Scalar multiplication
  static void Dot(cublasHandle_t cublas_handle, SHMatrix &A, 
                  SHMatrix &B, SHMatrix &C); //Matrix Dot Product

  // Returns pointer to data at desired location (GPU or CPU)
  static float* DataPointerAtLoc(SHMatrix& arg, mem_location desired_loc);

  void operator*=(SHMatrix &arg);
  void operator+=(SHMatrix &arg);
  void operator-=(SHMatrix &arg);
  void operator/=(SHMatrix &arg);

  void operator*=(float arg);
  void operator+=(float arg);
  void operator-=(float arg);
  void operator/=(float arg);

  cublasHandle_t cublas_handle;

  float *data;
  float scalar;
  float mean, mini, maxi;
  int mini_idx, maxi_idx;
  mem_location data_loc;
  std::vector<int> data_dims;
  int rows, cols, num_elems;
  std::string name;
  bool allocated;
  bool transpose_called, transpose_done;
  bool scale_called, scale_done;

  ~SHMatrix();

private:
  float alpha, beta; //dummy vars passed to some cuBLAS calls

  void load_dims(std::vector<int> &dims);
  void print_h_var(float *h_v, int r, int c, bool print_elem);
  void gaussian_init_gpu(float mean = 0.0f, float stddev = 0.1f);
  void gaussian_init_cpu(float mean = 0.0f, float stddev = 0.1f);
  void uniform_init_gpu(float lower = -0.5f, float higher = 0.5f);
  void uniform_init_cpu(float lower = -0.5f, float higher = 0.5f);

  void gpu2any_elemwise_mult(SHMatrix &arg);
  void gpu2any_elemwise_divide(SHMatrix &arg);
  void gpu2any_elemwise_add(SHMatrix &arg);
  void gpu2any_elemwise_subtract(SHMatrix &arg);
  static void gpu2any_dotproduct(cublasHandle_t cublas_handle, SHMatrix &A,
                                 SHMatrix &B, SHMatrix &C);

  void gpu2any_elemwise_op_worker(SHMatrix &arg, ELEM_OP elem_op);

  void gpu2any_elemwise_add(float arg);
  void gpu2any_elemwise_subtract(float arg);

  void cpu2any_elemwise_mult(SHMatrix &arg);
  void cpu2any_elemwise_add(SHMatrix &arg);
  void cpu2any_elemwise_subtract(SHMatrix &arg);
  void cpu2any_elemwise_divide(SHMatrix &arg);
  static void cpu2any_dotproduct(SHMatrix &A, SHMatrix &B, SHMatrix &C);

  void cpu2any_elemwise_op_worker(SHMatrix &arg, ELEM_OP elem_op);

  void cpu2any_elemwise_add(float arg);
  void cpu2any_elemwise_subtract(float arg);

  void duplicate_shmatrix(SHMatrix &src_shmatrix,
                          bool mem_alloc_needed = true);
  void copy_data_from(float *dst_ptr, float *src_ptr,
                      mem_location dst_loc, mem_location src_loc,
                      int copy_length);
  void deallocate_memory(float *mem_ptr, mem_location mem_loc);

  void transpose_worker_gpu(float coeff = 1.0f);
  void transpose_worker_ndim_gpu(float coeff = 1.0f);
  void transpose_worker_cpu(float coeff = 1.0f);

  void scale_worker_gpu(float coeff);
  void scale_worker_cpu(float coeff);
  
  void transpose_worker(float coeff = 1.0f);
  void scale_worker();

  int vect_to_lin_idx(std::vector<int> &vect_idx,
                      std::vector<int> &vect_dims);
  std::vector<int> lin_to_vect_idx(int lin_idx,
                                   std::vector<int> &vect_dims);

  void next_vect_idx(std::vector<int> &vect_idx,
                     std::vector<int> &vect_dims);

  void reset_metadata();
  float* allocate_memory();

  static bool transpose_decider(bool t_called, bool t_done);
  void init();

  void init_value_properties();
  void init_list_properties();
  void init_with_default_value(float *mem_ptr, mem_location mem_loc,
                               float init_val);
};