#!/bin/bash
NODES=$1
make clean
make all
mpirun -n $NODES -f $COBALT_NODEFILE ./run_mpi_test