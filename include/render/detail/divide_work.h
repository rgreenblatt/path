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

// assumes x dim, y dim are both greater than 32..
inline WorkDivision divide_work(unsigned samples_per, unsigned x_dim,
                                unsigned y_dim, unsigned target_block_size,
                                unsigned) {
#if 0
  unsigned target_work_per_block = target_block_size * target_work_per_thread;
#endif
  unsigned num_sample_blocks;
  unsigned sample_block_size;
  unsigned x_block_size;
  unsigned y_block_size;
  constexpr unsigned warp_size = 32;
  assert(target_block_size % warp_size == 0);
#if 0
  if (samples_per >= target_work_per_block) {
    num_sample_blocks = samples_per / target_work_per_block;
    x_block_size = 1;
    y_block_size = 1;
    unsigned divider = warp_size * num_sample_blocks;
    sample_block_size = ((samples_per + divider - 1) / divider) * warp_size;
  } else {
    num_sample_blocks = 1;
    x_block_size = target_work_per_block / samples_per;
    if (x_block_size >= warp_size) {
      y_block_size = x_block_size / warp_size;
      x_block_size = warp_size;
    } else {
      y_block_size = 1;
    }
    sample_block_size =
        ceil_divide(target_work_per_block, x_block_size * y_block_size);
  }
#else
  /* unsigned divisable_by = warp_size; */
  unsigned divisable_by = warp_size;

  if (samples_per > target_block_size) {
    divisable_by = target_block_size;
  }

  sample_block_size =
      ((samples_per + divisable_by - 1) / divisable_by) * divisable_by;
  num_sample_blocks = 1;
  x_block_size = 1;
  y_block_size = 1;
#endif

  return {std::min(sample_block_size, target_block_size),
          sample_block_size,
          x_block_size,
          y_block_size,
          num_sample_blocks,
          ceil_divide(x_dim, x_block_size),
          ceil_divide(y_dim, y_block_size)};
}
} // namespace detail
} // namespace render
