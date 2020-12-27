#pragma once

#include "lib/unit_vector.h"

#include <Eigen/Core>

namespace integrate {
namespace detail {
template <typename T> struct GenDirSample {
  UnitVector direction;
  T multiplier;
};
} // namespace detail

using FSample = detail::GenDirSample<float>;
using BSDFSample = detail::GenDirSample<Eigen::Array3f>;
} // namespace integrate
