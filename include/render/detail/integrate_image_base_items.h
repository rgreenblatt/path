#pragma once

#include "lib/bgra.h"
#include "lib/span.h"

#include <Eigen/Core>

namespace render {
namespace detail {
struct IntegrateImageBaseItems {
  bool output_as_bgra;
  unsigned samples_per;
  Span<BGRA> pixels;
  Span<Eigen::Array3f> intensities;
};
} // namespace detail
} // namespace render
