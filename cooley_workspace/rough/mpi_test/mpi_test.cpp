#include <iostream>
#include <mpi.h>


using namespace std;

int main() {
  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  std::cout << "Processor rank " << world_rank
    << " out of " << world_size << " nodes" << std::endl
    << "Processor: " << processor_name << std::endl;

  MPI_Finalize();
}
