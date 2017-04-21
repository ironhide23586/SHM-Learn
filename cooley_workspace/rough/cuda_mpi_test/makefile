CUDA_PATH=/home/souham/.local/cuda-8.0
MPI_PATH=/soft/libraries/mpi/mvapich2/gcc

# For Tesla K80-
COMPUTE_CAPABILITY=37

all: program

NAME=run_mpi_test
COMPILER=mpic++

cuda_mpi:
	nvcc -I$(MPI_PATH)/include -L$(MPI_PATH)/lib -lmpi mpi_test.cpp -o $(NAME)

program:
	$(COMPILER) mpi_test.cpp -std=c++11 -I$(CUDA_PATH)/include \
	-o $(NAME) -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -lcurand
	rm *.o

cudacode.o:
	nvcc -c -std=c++11 -I$(CUDA_PATH)/include -o mpi_testCU.o mpi_test.cpp \
	-gencode arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY)

clean:
	rm -rf *.o $(NAME)