#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <tuple>

namespace ray {
namespace detail {
inline HOST_DEVICE std::tuple<unsigned, unsigned>
get_block_idxs(unsigned general_block_idx, unsigned num_blocks_x) {
  unsigned block_idx_x = general_block_idx % num_blocks_x;
  unsigned block_idx_y = general_block_idx / num_blocks_x;

  return std::make_tuple(block_idx_x, block_idx_y);
}

inline unsigned num_blocks(unsigned size, unsigned block_size) {
  return (size + block_size - 1) / block_size;
};

struct BlockData {
  unsigned x_dim;
  unsigned y_dim;
  unsigned block_dim_x;
  unsigned block_dim_y;
  unsigned num_blocks_x;
  unsigned num_blocks_y;

  HOST_DEVICE std::tuple<unsigned, unsigned, unsigned, bool>
  getIndexes(SpanSized<const unsigned> group_indexes, unsigned block_idx,
             unsigned thread_idx) const {
    return getIndexes(group_indexes[block_idx], thread_idx);
  }

  HOST_DEVICE std::tuple<unsigned, unsigned, unsigned, bool>
  getIndexes(unsigned general_block_idx, unsigned thread_idx) const {
    auto [block_idx_x, block_idx_y] =
        get_block_idxs(general_block_idx, num_blocks_x);

    unsigned x = block_idx_x * block_dim_x + thread_idx % block_dim_x;
    unsigned y = block_idx_y * block_dim_y + thread_idx / block_dim_x;

    return std::make_tuple(x, y, x + y * x_dim, outsideBounds(x, y));
  }

  HOST_DEVICE unsigned generalNumBlocks() const {
    return num_blocks_x * num_blocks_y;
  }

  HOST_DEVICE unsigned generalBlockSize() const {
    return block_dim_x * block_dim_y;
  }

  HOST_DEVICE unsigned totalSize() const { return x_dim * y_dim; }

  HOST_DEVICE bool outsideBounds(unsigned x, unsigned y) const {
    return x >= x_dim || y >= y_dim;
  }

  BlockData(unsigned x_dim, unsigned y_dim, unsigned block_dim_x,
            unsigned block_dim_y)
      : x_dim(x_dim), y_dim(y_dim), block_dim_x(block_dim_x),
        block_dim_y(block_dim_y), num_blocks_x(num_blocks(x_dim, block_dim_x)),
        num_blocks_y(num_blocks(y_dim, block_dim_y)) {}

private:
  HOST_DEVICE std::tuple<unsigned, unsigned, unsigned, bool>
  generalGetIndexes(unsigned general_block_idx, unsigned thread_idx) const {
    auto [block_idx_x, block_idx_y] =
        get_block_idxs(general_block_idx, num_blocks_x);

    unsigned x = block_idx_x * block_dim_x + thread_idx % block_dim_x;
    unsigned y = block_idx_y * block_dim_y + thread_idx / block_dim_x;

    return std::make_tuple(x, y, x + y * x_dim, outsideBounds(x, y));
  }
};
} // namespace detail
} // namespace ray
