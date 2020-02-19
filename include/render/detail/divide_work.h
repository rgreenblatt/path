#pragma once

#include "lib/utils.h"

#include <assert.h>

namespace render {
namespace detail {
struct WorkDivision {
  unsigned blocks_per_pixel;
  unsigned samples_block_size;
  unsigned x_block_size;
  unsigned y_block_size;
};

// assumes x dim, y dim are both greater than 32..
inline WorkDivision divide_work(unsigned samples_per,
                                unsigned target_block_size,
                                unsigned target_work_per_thread) {
  unsigned target_work_per_block = target_block_size * target_work_per_thread;
  unsigned blocks_per_pixel;
  unsigned samples_block_size;
  unsigned x_block_size;
  unsigned y_block_size;
  constexpr unsigned warp_size = 32;
  if (samples_per >= target_work_per_block) {
    blocks_per_pixel = samples_per / target_work_per_block;
    x_block_size = 1;
    y_block_size = 1;
    unsigned divider = warp_size * blocks_per_pixel;
    samples_block_size = ((samples_per + divider - 1) / divider) * warp_size;
  } else {
    blocks_per_pixel = 1;
    x_block_size = target_work_per_block / samples_per;
    if (x_block_size >= warp_size) {
      y_block_size = x_block_size / warp_size;
      x_block_size = warp_size;
    } else {
      y_block_size = 1;
    }
    samples_block_size =
        ceil_divide(target_work_per_block, x_block_size * y_block_size);
  }
  assert(samples_block_size * blocks_per_pixel >= samples_per);
  assert(samples_block_size * x_block_size * y_block_size >=
         target_work_per_block);
  // TODO: check below assert...
  assert(samples_block_size * x_block_size * y_block_size <=
         2 * target_work_per_block);

  return {blocks_per_pixel, samples_block_size, x_block_size, y_block_size};
}
} // namespace detail
} // namespace render
