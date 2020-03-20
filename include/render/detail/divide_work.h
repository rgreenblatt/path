#pragma once

#include "lib/utils.h"

#include <assert.h>

namespace render {
namespace detail {
struct WorkDivision {
  unsigned block_size;
  unsigned sample_block_size;
  unsigned x_block_size;
  unsigned y_block_size;
  unsigned num_sample_blocks;
  unsigned num_x_blocks;
  unsigned num_y_blocks;
};

WorkDivision divide_work(unsigned samples_per, unsigned x_dim, unsigned y_dim,
                         unsigned target_block_size,
                         unsigned target_work_per_thread);
} // namespace detail
} // namespace render
