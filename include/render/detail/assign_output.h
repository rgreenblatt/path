#pragma once

#include "lib/cuda/utils.h"

#include "lib/bgra.h"
#include "lib/span.h"

#include <Eigen/Core>

namespace render {
namespace detail {
inline HOST_DEVICE void
assign_output(bool output_as_bgra, Span<BGRA> bgras,
              Span<Eigen::Array3f> intensities, unsigned sample_block_idx,
              unsigned num_sample_blocks, unsigned x, unsigned y,
              unsigned x_dim, unsigned samples_per, Eigen::Array3f intensity) {
  unsigned idx = sample_block_idx + num_sample_blocks * (x + y * x_dim);
  if (num_sample_blocks == 1) {
    intensity /= samples_per;
  }
  if (output_as_bgra && num_sample_blocks == 1) {
    bgras[idx] = intensity_to_bgr(intensity);
  } else {
    intensities[idx] = intensity;
  }
}
} // namespace detail
} // namespace render
