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
                                      unsigned y, Eigen::Array3f intensity) {
  unsigned idx =
      sample_block_idx + num_sample_blocks * (x + y * division.x_dim());
  if (num_sample_blocks == 1) {
    intensity /= b.samples_per;
  }
  if (b.output_as_bgra && num_sample_blocks == 1) {
    b.pixels[idx] = intensity_to_bgr(intensity);
  } else {
    b.intensities[idx] = intensity;
  }
}
} // namespace detail
} // namespace render
