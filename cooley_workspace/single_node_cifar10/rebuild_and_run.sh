#!/bin/bash
NODES=$1
make clean
make all
mv shmlearn_results.txt shmlearn_results_prev.txt
./convnet