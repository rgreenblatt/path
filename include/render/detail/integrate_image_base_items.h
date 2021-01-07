#pragma once

#include "lib/bgra.h"
#include "lib/span.h"
#include "work_division/work_division.h"

#include <Eigen/Core>

namespace render {
namespace detail {
using work_division::WorkDivision;
struct IntegrateImageBaseItems {
  bool output_as_bgra;
  unsigned samples_per;
  unsigned x_dim;
  unsigned y_dim;
  WorkDivision division;
  Span<BGRA> pixels;
  Span<Eigen::Array3f> intensities;
};
} // namespace detail
} // namespace render
