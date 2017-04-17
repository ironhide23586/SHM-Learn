#include <iostream>
#include <string>
#include <stdlib.h>
#include <mpi.h>

#define BCAST_SIZE 35
#define PROBED_RANK 3

using namespace std;

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

int main() {
  MPI_Init(NULL, NULL);

  float *data = (float *) malloc(BCAST_SIZE * sizeof(float));

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
    for (int i = 0; i < BCAST_SIZE; i++) {
      data[i] = i * 0.1f;
    }
    std::cout << "Data broadcasted -" << std::endl;
    print_h_var3(data, 1, BCAST_SIZE);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (my_rank == PROBED_RANK) {
    std::cout << "\nBefore Broadcast, Rank " << my_rank << std::endl;
    print_h_var3(data, 1, BCAST_SIZE);
  }

  MPI_Bcast(data, BCAST_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (my_rank == PROBED_RANK) {
    std::cout << "\nAfter Broadcast, Rank " << my_rank << std::endl;
    print_h_var3(data, 1, BCAST_SIZE);
  }

  MPI_Finalize();
  return 0;
}
