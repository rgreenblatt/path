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

#define CUDA_ERROR_CHK(ans)                                                    \
  { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "cuda assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#if 1
#ifdef __CUDACC__
#define ALIGN_STRUCT(x) __align__(x)
#else
#define ALIGN_STRUCT(x) alignas(x)
#endif
#endif
