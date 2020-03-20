#include "render/detail/divide_work.h"
#include "lib/utils.h"

#include "lib/info/debug_print.h"

#include <assert.h>

namespace render {
namespace detail {

constexpr unsigned make_divisable(unsigned v, unsigned divisable_by) {
  return ((v + divisable_by - 1) / divisable_by) * divisable_by;
}

// Testing:
static_assert(make_divisable(3, 1) == 3);
static_assert(make_divisable(4, 1) == 4);
static_assert(make_divisable(4, 2) == 4);
static_assert(make_divisable(4, 3) == 6);
static_assert(make_divisable(3, 3) == 3);
static_assert(make_divisable(100, 3) == 102);

WorkDivision divide_work(unsigned samples_per, unsigned x_dim, unsigned y_dim) {
  constexpr unsigned block_size = 256;
  constexpr unsigned x_block_size_thread = 32;
  constexpr unsigned y_block_size_thread = 8;

  static_assert(x_block_size_thread * y_block_size_thread == block_size);

  // SPEED: tune?
  // SPEED: Maybe it would be better to always just use thread???
  constexpr unsigned target_samples_per_thread = 8;
  constexpr unsigned warp_size = 32;
  /* constexpr unsigned half_target_samples_per_thread = */
  /*     std::max(target_samples_per_thread / 2, 1u); */

  static_assert(block_size % warp_size == 0);

  constexpr unsigned warps_per_block = block_size / warp_size;

  const unsigned num_sample_blocks = 1;

  ReductionStrategy sample_reduction_strategy;
  unsigned samples_per_thread;
  unsigned x_block_size;
  unsigned y_block_size;
  
  // This is going to have poor performance for large numbers of samples and
  // very few pixels
  if (samples_per >= block_size * target_samples_per_thread) {
    sample_reduction_strategy = ReductionStrategy::Block;
    unsigned new_samples_per = make_divisable(samples_per, block_size);
    samples_per_thread = new_samples_per / block_size;
    x_block_size = 1;
    y_block_size = 1;
  } else if (samples_per >= warp_size * target_samples_per_thread) {
    sample_reduction_strategy = ReductionStrategy::Warp;
    unsigned new_samples_per = make_divisable(samples_per, warp_size);
    samples_per_thread = new_samples_per / warp_size;
    x_block_size = warps_per_block;
    y_block_size = 1;
  } else {
    sample_reduction_strategy = ReductionStrategy::Thread;
    samples_per_thread = samples_per;
    x_block_size = x_block_size_thread;
    y_block_size = y_block_size_thread;
  }

  unsigned num_x_blocks = ceil_divide(x_dim, x_block_size);
  unsigned num_y_blocks = ceil_divide(y_dim, y_block_size);

  return {
      sample_reduction_strategy,
      block_size,
      samples_per_thread,
      x_block_size,
      y_block_size,
      num_sample_blocks,
      num_x_blocks,
      num_y_blocks,
  };
}
} // namespace detail
} // namespace render
