#pragma once

#include "kernel/work_division.h"
#include "lib/cuda/utils.h"
#include "meta/specialization_of.h"
#include "render/detail/integrate_image/base_items.h"

namespace render {
namespace detail {
namespace integrate_image {
inline HOST_DEVICE void assign_output(const BaseItems &b, unsigned x_dim,
                                      unsigned sample_block_idx,
                                      unsigned num_sample_blocks, unsigned x,
                                      unsigned y, FloatRGB float_rgb) {
  unsigned idx = sample_block_idx + num_sample_blocks * (x + y * x_dim);
  if (num_sample_blocks == 1) {
    float_rgb /= b.samples_per;
  }
  if (b.output_as_bgra_32 && num_sample_blocks == 1) {
    b.bgra_32[idx] = float_rgb_to_bgra_32(float_rgb);
  } else {
    b.float_rgb[idx] = float_rgb;
  }
}

// assign when no additional reduction will be required
inline HOST_DEVICE void assign_output_single(const BaseItems &b, unsigned x_dim,
                                             unsigned x, unsigned y,
                                             FloatRGB float_rgb) {
  assign_output(b, x_dim, 0, 1, x, y, float_rgb);
}
} // namespace integrate_image
} // namespace detail
} // namespace render
