#include <iostream>
#include <string>
#include <stdlib.h>
#include <mpi.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BCAST_SIZE 6
#define GATHER_CHUNK 3
#define PROBED_RANK 3

using namespace std;


void print_h_var3(float *h_v, int r, int c) {
  std::cout << "\n-------------------------" << std::endl;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      std::cout << h_v[j + i * c] << "\t";
    }
    std::cout << std::endl;
  }
}

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

// __global__ void init_GPU_mat(float *d_mat, int sz) {
//   int idx = blockDim.x * blockIdx.x + threadIdx.x;
//   d_mat[idx] = idx * 0.1f;
// }

int main() {
  MPI_Init(NULL, NULL);

  // float *data = (float *)malloc(BCAST_SIZE * sizeof(float));
  // float *gathered_data;
  // float *gather_part = (float *)malloc(GATHER_CHUNK * sizeof(float));

  float *h_data, *d_data, *d_gathered_data, *h_gather_part, *d_gather_part;
  cudaMalloc((void **)&d_data, BCAST_SIZE * sizeof(float));
  h_data = (float *)malloc(BCAST_SIZE * sizeof(float));
  cudaMalloc((void **)&d_gather_part, GATHER_CHUNK * sizeof(float));
  h_gather_part = (float *)malloc(GATHER_CHUNK * sizeof(float));

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  std::cout << "Processor rank " << my_rank
    << " out of " << world_size << " nodes" << std::endl
    << "Processor: " << processor_name << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  if (my_rank == 0) {
    //init_GPU_mat<<<1, BCAST_SIZE>>>(d_data, BCAST_SIZE);
    for (int i = 0; i < BCAST_SIZE; i++) {
      h_data[i] = i * 0.1f;
    }
    cudaMemcpy(d_data, h_data, BCAST_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Data broadcasted -" << std::endl;
    print_d_var3(d_data, 1, BCAST_SIZE);
    //print_h_var3(data, 1, BCAST_SIZE);
    // gathered_data = (float *)malloc(GATHER_CHUNK * world_size * sizeof(float));
    cudaMalloc((void **)&d_gathered_data, GATHER_CHUNK * world_size * sizeof(float));
  }

  for (int i = my_rank * GATHER_CHUNK; i < (my_rank + 1) * GATHER_CHUNK; i++) {
    h_gather_part[i - (my_rank * GATHER_CHUNK)] = i;
  }

  cudaMemcpy(d_gather_part, h_gather_part, GATHER_CHUNK * sizeof(float), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < world_size; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == i) {
      std::cout << "\nBefore Broadcast, Rank " << my_rank << std::endl;
      //print_h_var3(data, 1, BCAST_SIZE);
      print_d_var3(d_data, 1, BCAST_SIZE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0)
    std::cout << "Broadcasting..." << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(d_data, BCAST_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0)
    std::cout << "Broadcast complete" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < world_size; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == i) {
      std::cout << "\nAfter Broadcast, Rank " << my_rank << std::endl;
      //print_h_var3(data, 1, BCAST_SIZE);
      print_d_var3(d_data, 1, BCAST_SIZE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0) {
    std::cout << "\nBefore Gather -" << std::endl;
    print_d_var3(d_gathered_data, 1, world_size * GATHER_CHUNK);
  }

  MPI_Gather(d_gather_part, GATHER_CHUNK, MPI_FLOAT, d_gathered_data, GATHER_CHUNK,
             MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0) {
    std::cout << "\nAfter Gather -" << std::endl;
    print_d_var3(d_gathered_data, 1, world_size * GATHER_CHUNK);
  }

  MPI_Finalize();
  return 0;
}
