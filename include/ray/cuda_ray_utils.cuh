#pragma once

#include "ray/ray_utils.h"

namespace ray {
namespace detail {
__inline__ __device__ std::tuple<unsigned, unsigned>
get_block_idxs(unsigned general_block_idx, unsigned num_blocks_x) {
  unsigned block_idx_x = general_block_idx % num_blocks_x;
  unsigned block_idx_y = general_block_idx / num_blocks_x;

  return std::make_tuple(block_idx_x, block_idx_y);
}

__inline__ __device__ std::tuple<unsigned, unsigned>
get_indexes(const unsigned *group_indexes, bool is_sparse,
            unsigned num_blocks_x, unsigned block_dim_x, unsigned block_dim_y) {
  unsigned general_block_idx =
      is_sparse ? group_indexes[blockIdx.x] : blockIdx.x;

  auto [block_idx_x, block_idx_y] =
      get_block_idxs(general_block_idx, num_blocks_x);

  unsigned x = block_idx_x * block_dim_x + threadIdx.x % block_dim_x;
  unsigned y = block_idx_y * block_dim_y + threadIdx.x / block_dim_x;

  return std::make_tuple(x, y);
}

__inline__ __device__ std::tuple<unsigned, unsigned>
get_non_sparse_indexes(unsigned num_blocks_x, unsigned block_dim_x,
                       unsigned block_dim_y) {
  return get_indexes(nullptr, false, num_blocks_x, block_dim_x, block_dim_y);
}

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
