#pragma once

#include "lib/float_rgb.h"
#include "lib/unit_vector.h"

namespace integrate {
namespace detail {
template <typename T> struct GenDirSample {
  UnitVector direction;
  T multiplier;
};
} // namespace detail

using FSample = detail::GenDirSample<float>;
using BSDFSample = detail::GenDirSample<FloatRGB>;
} // namespace integrate
