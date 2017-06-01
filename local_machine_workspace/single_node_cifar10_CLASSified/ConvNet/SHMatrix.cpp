#include "SHMatrix.h"

SHMatrix::SHMatrix(const cublasHandle_t &cublas_handle_arg,
                   float *mat_data, std::vector<int> &dims,
                   mem_location loc)
  : cublas_handle(cublas_handle_arg),
    data(mat_data),
    data_dims(dims),
    data_loc(loc),
    allocated(false) {
  transpose_called = false;
  transpose_done = false;
  scale_called = false;
  scale_done = false;
  load_dims(dims);
  allocated = true;
}

SHMatrix::SHMatrix(const cublasHandle_t &cublas_handle_arg,
                   std::vector<int> &dims,
                   mem_location loc, bool default_init,
                   float init_val)
  : cublas_handle(cublas_handle_arg), 
    data_dims(dims),
    data_loc(loc),
    allocated(false) {
  transpose_called = false;
  transpose_done = false;
  scale_called = false;
  scale_done = false;
  load_dims(dims);
  allocate_memory();
  if (data_loc == GPU) {
    if (default_init) {
      FloatCUDAMemset(data, num_elems, init_val);
      CudaCheckError();
    }
  }
  else if (data_loc == CPU) {
    if (default_init) {
      for (int i = 0; i < num_elems; i++) {
        data[i] = init_val;
      }
    }
  }
}

SHMatrix::SHMatrix(const cublasHandle_t &cublas_handle_arg,
                   SHMatrix &src_shmatrix, mem_location loc)
  : cublas_handle(cublas_handle_arg),
    data_loc(loc), allocated(false) {
  transpose_called = false;
  transpose_done = false;
  scale_called = false;
  scale_done = false;
  duplicate_shmatrix(src_shmatrix);
}

void SHMatrix::Equate(SHMatrix &src_shmatrix) {
  if (allocated)
    Clear();
  duplicate_shmatrix(src_shmatrix);
}

void SHMatrix::Print(bool print_elems) {
  float *h_v;
  if (data_loc == GPU) {
    h_v = (float *)malloc(sizeof(float) * rows * cols);
    CudaSafeCall(cudaMemcpy(h_v, data, sizeof(float) * rows * cols,
                            cudaMemcpyDeviceToHost));
  }
  else if (data_loc == CPU) {
    h_v = data;
  }
  print_h_var(h_v, rows, cols, print_elems);
  if (data_loc == GPU)
    free(h_v);
}

void SHMatrix::Move2GPU() {
  if (data_loc == GPU)
    return;
  float *d_data;
  CudaSafeCall(cudaMalloc((void **)&d_data, sizeof(float) * num_elems));
  CudaSafeCall(cudaMemcpy(d_data, data, sizeof(float) * num_elems,
                          cudaMemcpyHostToDevice));
  free(data);
  data = d_data;
  data_loc = GPU;
}

void SHMatrix::Move2CPU() {
  if (data_loc == CPU)
    return;
  float *h_data = (float *)malloc(sizeof(float) * num_elems);
  CudaSafeCall(cudaMemcpy(h_data, data, sizeof(float) * num_elems,
                          cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(data));
  data = h_data;
  data_loc = CPU;
}

void SHMatrix::Clear() {
  if (data_loc == GPU) {
    if (allocated)
      CudaSafeCall(cudaFree(data));
  }
  else if (data_loc == CPU) {
    if (allocated)
      free(data);
  }
  reset_metadata();
}

void SHMatrix::GaussianInit(float mean, float stddev) {
  if (data_loc == GPU) {
    gaussian_init_gpu(mean, stddev);
  }
  else if (data_loc == CPU) {
    gaussian_init_cpu(mean, stddev);
  }
  allocated = true;
}

void SHMatrix::UniformInit(float mean, float stddev) {
  if (data_loc == GPU) {
    uniform_init_gpu(mean, stddev);
  }
  else if (data_loc == CPU) {
    uniform_init_cpu(mean, stddev);
  }
  allocated = true;
}

float SHMatrix::GetGaussianNum(float mean, float stddev) {
  static std::default_random_engine re;
  static std::normal_distribution<float> dist(mean, stddev);
  return dist(re);
}

float SHMatrix::GetUniformNum(float lower, float higher) {
  static std::default_random_engine re;
  static std::uniform_real_distribution<float> dist(lower, higher);
  return dist(re);
}

// Transpose operation : speeds up computation by postponing T operations
void SHMatrix::T() { //mini_idx & maxi_idx computation pending
  if (!transpose_done)
    transpose_called = ~transpose_called;
  else
    transpose_called = true;
  transpose_done = false;
  int tmp = cols;
  cols = rows;
  rows = tmp;
  std::reverse(data_dims.begin(), data_dims.end());
}

void SHMatrix::operator*=(SHMatrix &arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_mult(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_mult(arg);
  }
}

void SHMatrix::operator+=(SHMatrix &arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_add(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_add(arg);
  }
}

void SHMatrix::operator-=(SHMatrix &arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_subtract(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_subtract(arg);
  }
}

void SHMatrix::operator/=(SHMatrix &arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_divide(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_divide(arg);
  }
}

void SHMatrix::operator*=(float arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_mult(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_mult(arg);
  }
}

void SHMatrix::operator+=(float arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_add(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_add(arg);
  }
}

void SHMatrix::operator-=(float arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_subtract(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_subtract(arg);
  }
}

void SHMatrix::operator/=(float arg) {
  if (data_loc == GPU) {
    gpu2any_elemwise_divide(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_divide(arg);
  }
}

SHMatrix SHMatrix::operator*(SHMatrix &arg) {
  return arg;
}

SHMatrix SHMatrix::operator+(SHMatrix &arg) {
  return arg;
}

SHMatrix SHMatrix::operator-(SHMatrix &arg) {
  return arg;
}

SHMatrix SHMatrix::operator/(SHMatrix &arg) {
  return arg;
}

void SHMatrix::gaussian_init_gpu(float mean, float stddev) {
  curandGenerator_t rng;
  CurandSafeCall(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW));
  CurandSafeCall(curandGenerateNormal(rng, data, sizeof(float) * num_elems,
                                      mean, stddev));
  CurandSafeCall(curandDestroyGenerator(rng));
}

void SHMatrix::gaussian_init_cpu(float mean, float stddev) {
  for (int i = 0; i < num_elems; i++) {
    data[i] = GetGaussianNum(mean, stddev);
  }
}

void SHMatrix::uniform_init_gpu(float lower, float higher) {
  curandGenerator_t rng;
  CurandSafeCall(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW));
  CurandSafeCall(curandGenerateUniform(rng, data, sizeof(float) * num_elems));
  CurandSafeCall(curandDestroyGenerator(rng));
  ScaleUniformSHMatrix(data, num_elems, lower, higher);
}

void SHMatrix::uniform_init_cpu(float lower, float higher) {
  for (int i = 0; i < num_elems; i++) {
    data[i] = GetUniformNum(lower, higher);
  }
}

SHMatrix::~SHMatrix() { Clear(); }

void SHMatrix::load_dims(std::vector<int> &dims) {
  cols = 1;
  rows = dims[0];
  for (int i = 1; i < dims.size(); i++) {
    cols *= dims[i];
  }
  num_elems = rows * cols;
}

void SHMatrix::print_h_var(float *h_v, int r, int c, bool print_elem) {
  std::cout << "-------------------------" << std::endl;
  if (!allocated) {
    std::cout << "<NULL_SHMATRIX>" << std::endl;
  }
  mini = h_v[0];
  maxi = h_v[0];
  float sum = 0.0f;
  mini_idx = 0;
  maxi_idx = 0;
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
  mean = sum / (r * c);
  if (allocated) {
    std::cout << "Shape = (";
    for (int i = 0; i < data_dims.size() - 1; i++) {
      std::cout << data_dims[i] << ", ";
    }
    std::cout << data_dims[data_dims.size() - 1] << ")" << std::endl;
    std::cout << "Minimum at index " << mini_idx << " = " << mini << std::endl;
    std::cout << "Maximum at index " << maxi_idx << " = " << maxi << std::endl;
    std::cout << "Average of all elements = " << mean << std::endl;
  }
  std::cout << "Number of elements = " << num_elems << std::endl;
  if (data_loc == GPU) {
    std::cout << "Location = GPU" << std::endl;
  }
  else if (data_loc == CPU) {
    std::cout << "Location = CPU" << std::endl;
  }
}

void SHMatrix::gpu2any_elemwise_mult(SHMatrix &arg) {
  float *d_arg_data;
  if (arg.data_loc == GPU) {
    d_arg_data = arg.data;
  }
  else if (arg.data_loc == CPU) {
    CudaSafeCall(cudaMalloc((void **)&d_arg_data,
                            sizeof(float) * arg.num_elems));
    CudaSafeCall(cudaMemcpy(d_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyHostToDevice));
  }
  ElemwiseMultiplyInPlaceGPU(data, d_arg_data, num_elems);
  if (arg.data_loc == CPU) {
    CudaSafeCall(cudaFree(d_arg_data));
  }
}

void SHMatrix::gpu2any_elemwise_add(SHMatrix &arg) {
  float *d_arg_data;
  if (arg.data_loc == GPU) {
    d_arg_data = arg.data;
  }
  else if (arg.data_loc == CPU) {
    CudaSafeCall(cudaMalloc((void **)&d_arg_data,
                            sizeof(float) * arg.num_elems));
    CudaSafeCall(cudaMemcpy(d_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyHostToDevice));
  }
  ElemwiseAddInPlaceGPU(data, d_arg_data, num_elems);
  //CublasSafeCall(cublasAxpyEx(cublas_handle,
  if (arg.data_loc == CPU) {
    CudaSafeCall(cudaFree(d_arg_data));
  }
}

void SHMatrix::gpu2any_elemwise_subtract(SHMatrix &arg) {
  float *d_arg_data;
  if (arg.data_loc == GPU) {
    d_arg_data = arg.data;
  }
  else if (arg.data_loc == CPU) {
    CudaSafeCall(cudaMalloc((void **)&d_arg_data,
                            sizeof(float) * arg.num_elems));
    CudaSafeCall(cudaMemcpy(d_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyHostToDevice));
  }
  ElemwiseSubtractInPlaceGPU(data, d_arg_data, num_elems);
  if (arg.data_loc == CPU) {
    CudaSafeCall(cudaFree(d_arg_data));
  }
}

void SHMatrix::gpu2any_elemwise_divide(SHMatrix &arg) {
  float *d_arg_data;
  if (arg.data_loc == GPU) {
    d_arg_data = arg.data;
  }
  else if (arg.data_loc == CPU) {
    CudaSafeCall(cudaMalloc((void **)&d_arg_data,
                            sizeof(float) * arg.num_elems));
    CudaSafeCall(cudaMemcpy(d_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyHostToDevice));
  }
  ElemwiseDivideInPlaceGPU(data, d_arg_data, num_elems);
  if (arg.data_loc == CPU) {
    CudaSafeCall(cudaFree(d_arg_data));
  }
}

void SHMatrix::gpu2any_elemwise_mult(float arg) {

}

void SHMatrix::gpu2any_elemwise_add(float arg) {

}

void SHMatrix::gpu2any_elemwise_subtract(float arg) {

}

void SHMatrix::gpu2any_elemwise_divide(float arg) {

}

void SHMatrix::cpu2any_elemwise_mult(SHMatrix &arg) {
  float *h_arg_data;
  if (arg.data_loc == GPU) {
    h_arg_data = (float *)malloc(sizeof(float)
                                 * arg.num_elems);
    CudaSafeCall(cudaMemcpy(h_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyDeviceToHost));
  }
  else if (arg.data_loc == CPU) {
    h_arg_data = arg.data;
  }
  ElemwiseMultiplyInPlaceCPU(data, h_arg_data, num_elems);
  if (arg.data_loc == GPU) {
    free(h_arg_data);
  }
}

void SHMatrix::cpu2any_elemwise_add(SHMatrix &arg) {
  float *h_arg_data;
  if (arg.data_loc == GPU) {
    h_arg_data = (float *)malloc(sizeof(float)
                                 * arg.num_elems);
    CudaSafeCall(cudaMemcpy(h_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyDeviceToHost));
  }
  else if (arg.data_loc == CPU) {
    h_arg_data = arg.data;
  }
  ElemwiseAddInPlaceCPU(data, h_arg_data, num_elems);
  if (arg.data_loc == GPU) {
    free(h_arg_data);
  }
}

void SHMatrix::cpu2any_elemwise_subtract(SHMatrix &arg) {
  float *h_arg_data;
  if (arg.data_loc == GPU) {
    h_arg_data = (float *)malloc(sizeof(float)
                                 * arg.num_elems);
    CudaSafeCall(cudaMemcpy(h_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyDeviceToHost));
  }
  else if (arg.data_loc == CPU) {
    h_arg_data = arg.data;
  }
  ElemwiseSubtractInPlaceCPU(data, h_arg_data, num_elems);
  if (arg.data_loc == GPU) {
    free(h_arg_data);
  }
}

void SHMatrix::cpu2any_elemwise_divide(SHMatrix &arg) {
  float *h_arg_data;
  if (arg.data_loc == GPU) {
    h_arg_data = (float *)malloc(sizeof(float)
                                 * arg.num_elems);
    CudaSafeCall(cudaMemcpy(h_arg_data, arg.data,
                            sizeof(float) * arg.num_elems,
                            cudaMemcpyDeviceToHost));
  }
  else if (arg.data_loc == CPU) {
    h_arg_data = arg.data;
  }
  ElemwiseDivideInPlaceCPU(data, h_arg_data, num_elems);
  if (arg.data_loc == GPU) {
    free(h_arg_data);
  }
}

void SHMatrix::cpu2any_elemwise_mult(float arg) {

}

void SHMatrix::cpu2any_elemwise_add(float arg) {

}

void SHMatrix::cpu2any_elemwise_subtract(float arg) {

}

void SHMatrix::cpu2any_elemwise_divide(float arg) {

}

void SHMatrix::duplicate_shmatrix(SHMatrix &src_shmatrix) {
  for (int i = 0; i < src_shmatrix.data_dims.size(); i++) {
    data_dims.push_back(src_shmatrix.data_dims[i]);
  }
  load_dims(data_dims);
  allocate_memory();
  copy_data_from(src_shmatrix);
}

void SHMatrix::copy_data_from(SHMatrix &src_shmatrix) {
  if (src_shmatrix.data_loc == GPU) {
    if (data_loc == GPU) {
      CudaSafeCall(cudaMemcpy(data, src_shmatrix.data, 
                              sizeof(float) * num_elems,
                              cudaMemcpyDeviceToDevice));
    }
    else if (data_loc == CPU) {
      CudaSafeCall(cudaMemcpy(data, src_shmatrix.data, 
                              sizeof(float) * num_elems,
                              cudaMemcpyDeviceToHost));
    }
  }
  else if (src_shmatrix.data_loc == CPU) {
    if (data_loc == GPU) {
      CudaSafeCall(cudaMemcpy(data, src_shmatrix.data, 
                              sizeof(float) * num_elems,
                              cudaMemcpyHostToDevice));
    }
    else if (data_loc == CPU) {
      memcpy(data, src_shmatrix.data, sizeof(float) * num_elems);
    }
  }
}

void SHMatrix::reset_metadata() {
  if (allocated) {
    mean = 0.0f;
    mini = 0.0f;
    maxi = 0.0f;
    data_dims.clear();
    name.clear();
    rows = 0;
    cols = 0;
    num_elems = 0;
    mini_idx = 0;
    maxi_idx = 0;
    transpose_called = false;
    transpose_done = false;
    scale_called = false;
    scale_done = false;
    allocated = false;
  }
}

void SHMatrix::allocate_memory() {
  if (data_loc == GPU) {
    CudaSafeCall(cudaMalloc((void **)&data, sizeof(float) * num_elems));
  }
  else if (data_loc == CPU) {
    data = (float *)malloc(sizeof(float) * num_elems);
  }
  allocated = true;
}