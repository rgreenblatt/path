#pragma once

#include <cstdio>
#include <cstdlib>

// needed for some reason when using clang...
#define CUB_USE_COOPERATIVE_GROUPS

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
// wrap to avoid language server error...
#define syncthreads_wrapped() __syncthreads()
#else
#include <cuda_runtime.h>
#define HOST_DEVICE
#define DEVICE
// wrap to avoid language server error...
#define syncthreads_wrapped()
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

inline constexpr unsigned warp_size = 32;
inline constexpr unsigned max_num_warps_per_block = 32;
