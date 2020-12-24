#pragma once

#include "bsdf/bsdf.h"

#include <Eigen/Core>

namespace bsdf {
template <BSDF T> struct Material {
  using BSDFT = T;
  T bsdf;
  Eigen::Array3f emission;
};
} // namespace bsdf
