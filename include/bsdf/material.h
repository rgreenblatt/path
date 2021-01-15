#pragma once

#include "bsdf/bsdf.h"
#include "lib/float_rgb.h"

namespace bsdf {
template <BSDF T> struct Material {
  using BSDFT = T;
  T bsdf;
  FloatRGB emission;
};
} // namespace bsdf
