#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>

#include <iostream>
#include <random>
#include <chrono>

using namespace std;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

// #define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
// #define CudnnSafeCall( err ) __cudnnSafeCall( err, __FILE__, __LINE__ )
// #define CublasSafeCall( err ) __cublasSafeCall( err, __FILE__, __LINE__ )
// #define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


namespace NeuralNet {

  class ErrorChecker {
  public:
    const char* cublasGetErrorString(cublasStatus_t error);
    void __cudaSafeCall(cudaError err, const char *file, const int line);
    void __cudnnSafeCall(cudnnStatus_t err, const char *file, const int line);
    void __cublasSafeCall(cublasStatus_t err, const char *file, const int line);
    void __cudaCheckError(const char *file, const int line);
  };
}