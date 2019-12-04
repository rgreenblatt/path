#pragma once

#include "ray/ray_utils.h"

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
