#pragma once

#include "kernel/work_division.h"
#include "lib/cuda/utils.h"
#include "meta/specialization_of.h"
#include "render/detail/integrate_image_base_items.h"

namespace render {
namespace detail {
inline HOST_DEVICE void assign_output(const IntegrateImageBaseItems &b,
                                      const kernel::WorkDivision &division,
                                      unsigned sample_block_idx,
                                      unsigned num_sample_blocks, unsigned x,
                                      unsigned y, FloatRGB float_rgb) {
  unsigned idx =
      sample_block_idx + num_sample_blocks * (x + y * division.x_dim());
  if (num_sample_blocks == 1) {
    float_rgb /= b.samples_per;
  }
  if (b.output_as_bgra_32 && num_sample_blocks == 1) {
    b.bgra_32[idx] = float_rgb_to_bgra_32(float_rgb);
  } else {
    b.float_rgb[idx] = float_rgb;
  }
}
} // namespace detail
} // namespace render
