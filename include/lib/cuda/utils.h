#pragma once

#ifdef __CUDACC__
#include "__clang_cuda_runtime_wrapper.h"
#endif

#include <stdio.h>
#include <stdlib.h>

// needed for some reason when using clang...
#define CUB_USE_COOPERATIVE_GROUPS

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#include <cuda_runtime.h>
#define HOST_DEVICE
#endif

inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "cuda assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#define CUDA_ERROR_CHK(ans)                                                    \
  { cuda_assert((ans), __FILE__, __LINE__); }

#if 0
static const char* curand_get_error_string(curandStatus_t error)
{
    switch (error)
    {
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

inline void curand_assert(curandStatus_t code, const char *file, int line) {
  if (code != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "curand assert: %s %s %d\n", curand_get_error_string(code), file,
            line);
    exit(code);
  }
}

#define CURAND_ERROR_CHK(ans)                                                  \
  { curand_assert((ans), __FILE__, __LINE__); }
#endif

#if 1
#ifdef __CUDACC__
#define ALIGN_STRUCT(x) __align__(x)
#else
#define ALIGN_STRUCT(x) alignas(x)
#endif
#endif
