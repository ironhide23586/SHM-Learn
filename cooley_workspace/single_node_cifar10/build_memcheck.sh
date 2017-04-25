#!/bin/bash
make clean
make all
cuda-memcheck convnet > memcheck_results.txt