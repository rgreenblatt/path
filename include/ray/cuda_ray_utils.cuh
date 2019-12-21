#pragma once

#include "ray/ray_utils.h"

namespace ray {
namespace detail {
constexpr uint32_t full_mask = 0xffffffff;

__inline__ __device__ uint8_t warp_reduce_cond(uint8_t val) {
  for (unsigned offset = warpSize / 2; offset > 0; offset /= 2) {
    val = __shfl_down_sync(full_mask, val, offset) && val;
  }

  return val;
}

__inline__ __device__ uint8_t block_reduce_cond(uint8_t val,
                                                unsigned thread_block_index,
                                                unsigned thread_block_size) {
  static __shared__ uint8_t shared[32]; // Shared mem for 32 partial sums

  unsigned lane = thread_block_index % warpSize;
  unsigned wid = thread_block_index / warpSize;

  val = warp_reduce_cond(val); // Each warp performs partial reduction

  if (lane == 0) {
    shared[wid] = val; // Write reduced value to shared memory
  }

  __syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (thread_block_index < thread_block_size / warpSize) ? shared[lane] : 1;

  val = warp_reduce_cond(val); // Final reduce within first warp

  return val;
}
} // namespace detail
} // namespace ray
