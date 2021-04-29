#pragma once

#include "lib/bgra_32.h"
#include "lib/float_rgb.h"
#include "lib/span.h"

namespace render {
namespace detail {
namespace integrate_image {
struct BaseItems {
  bool output_as_bgra_32;
  unsigned samples_per;
  Span<BGRA32> bgra_32;
  Span<FloatRGB> float_rgb;
};
} // namespace integrate_image
} // namespace detail
} // namespace render
