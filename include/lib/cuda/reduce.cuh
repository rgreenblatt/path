#pragma once

#include "lib/cuda/utils.h"
#include <cstdint>

#ifdef __CUDACC__ // To avoid issues with the language server
constexpr uint32_t full_mask = 0xffffffff;

template <typename T, typename F>
inline __device__ T warp_reduce(T val, const F &f) {
  for (unsigned offset = warp_size / 2; offset > 0; offset /= 2) {
    val = f(__shfl_down_sync(full_mask, val, offset), val);
  }

  return val;
}

template <typename T, typename F>
__inline__ __device__ T block_reduce(T val, const F &f, const T &identity_value,
                                     unsigned thread_block_index,
                                     unsigned thread_block_size) {
  static __shared__ T shared[32]; // Shared mem for 32 partial sums

  unsigned lane = thread_block_index % warp_size;
  unsigned wid = thread_block_index / warp_size;

  val = warp_reduce(val, f); // Each warp performs partial reduction

  if (lane == 0) {
    shared[wid] = val; // Write reduced value to shared memory
  }

  __syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (thread_block_index < thread_block_size / warp_size) ? shared[lane]
                                                             : identity_value;

  val = warp_reduce(val, f); // Final reduce within first warp

  return val;
}

__inline__ __device__ bool block_reduce_cond(bool val,
                                             unsigned thread_block_index,
                                             unsigned thread_block_size) {
  return block_reduce(
      val, [] __device__(bool first, bool second) { return first && second; },
      true, thread_block_index, thread_block_size);
}
#endif
