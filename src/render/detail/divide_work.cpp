#include "render/detail/divide_work.h"
#include "lib/utils.h"

#include <assert.h>

namespace render {
namespace detail {
WorkDivision divide_work(unsigned samples_per, unsigned x_dim, unsigned y_dim,
                         unsigned target_block_size, unsigned) {
  constexpr unsigned warp_size = 32;

  assert(target_block_size % warp_size == 0);

  unsigned divisable_by = warp_size;

  if (samples_per > target_block_size) {
    divisable_by = target_block_size;
  }

  unsigned sample_block_size =
      ((samples_per + divisable_by - 1) / divisable_by) * divisable_by;
  unsigned num_sample_blocks = 1;
  unsigned x_block_size = 1;
  unsigned y_block_size = 1;

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
