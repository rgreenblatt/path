#pragma once

#include "intersect/ray.h"
#include "lib/float_rgb.h"
#include "lib/optional.h"

namespace integrate {
namespace detail {
template <typename T> struct RayInfo {
  T multiplier;
  std::optional<float> target_distance;
};

template <typename T> struct RayRayInfo {
  intersect::Ray ray;
  RayInfo<T> info;
};
} // namespace detail

using FRayInfo = detail::RayInfo<float>;
using ArrRayInfo = detail::RayInfo<FloatRGB>;
using FRayRayInfo = detail::RayRayInfo<float>;
using ArrRayRayInfo = detail::RayRayInfo<FloatRGB>;
} // namespace integrate
