#pragma once

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudnnSafeCall( err ) __cudnnSafeCall( err, __FILE__, __LINE__ )
#define CublasSafeCall( err ) __cublasSafeCall( err, __FILE__, __LINE__ )
#define CurandSafeCall( err ) __curandSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}
#endif

#ifdef CURANDAPI
// cuRAND API errors
static const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}
#endif

inline void __cudaSafeCall(cudaError err,
                           const char *file,
                           const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

inline void __cudnnSafeCall(cudnnStatus_t err,
                            const char *file,
                            const int line) {
#ifdef CUDA_ERROR_CHECK
  if (CUDNN_STATUS_SUCCESS != err) {
    fprintf(stderr, "cudnnSafeCall() failed at %s:%i : %s\n",
            file, line, cudnnGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

inline void __cublasSafeCall(cublasStatus_t err,
                             const char *file,
                             const int line) {
#ifdef CUDA_ERROR_CHECK
  if (CUBLAS_STATUS_SUCCESS != err) {
    fprintf(stderr, "cublasSafeCall() failed at %s:%i : %s\n",
            file, line, cublasGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

inline void __curandSafeCall(curandStatus_t err,
                             const char *file,
                             const int line) {
#ifdef CUDA_ERROR_CHECK
  if (CURAND_STATUS_SUCCESS != err) {
    fprintf(stderr, "curandSafeCall() failed at %s:%i : %s\n",
            file, line, curandGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

inline void __cudaCheckError(const char *file,
                             const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}