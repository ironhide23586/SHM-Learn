#include "SHMatrix.h"

SHMatrix::SHMatrix(float *mat_data, std::vector<int> &dims,
                   mem_location loc)
  : data(mat_data),
    data_dims(dims),
    data_loc(loc) {
  load_dims(dims);
}

SHMatrix::SHMatrix(std::vector<int> &dims,
                   mem_location loc, bool default_init,
                   float init_val)
  : data_dims(dims),
    data_loc(loc) {
  load_dims(dims);
  if (data_loc == GPU) {
    CudaSafeCall(cudaMalloc((void **)&data, sizeof(float) * num_elems));
    if (default_init) {
      FloatCUDAMemset(data, num_elems, init_val);
      CudaCheckError();
    }
  }
  else if (data_loc == CPU) {
    data = (float *)malloc(sizeof(float) * num_elems);
    if (default_init) {
      for (int i = 0; i < num_elems; i++) {
        data[i] = init_val;
      }
    }
  }
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

SHMatrix::~SHMatrix() { }

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
  std::cout << "Shape = (" << r << ", " << c << ")" << std::endl;
  std::cout << "Minimum at index " << mini_idx << " = " << mini << std::endl;
  std::cout << "Maximum at index " << maxi_idx << " = " << maxi << std::endl;
  std::cout << "Average of all elements = " << mean << std::endl;
  if (data_loc == GPU) {
    std::cout << "Location = GPU" << std::endl;
  }
  else if (data_loc == CPU) {
    std::cout << "Location = CPU" << std::endl;
  }
}