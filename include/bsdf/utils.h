#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/unit_vector.h"

#include <Eigen/Geometry>

namespace bsdf {
ATTR_PURE_NDEBUG HOST_DEVICE inline UnitVector
reflect_over_normal(const UnitVector &vec, const UnitVector &normal) {
  return UnitVector::new_normalize(*vec + 2.0f * -vec->dot(*normal) * *normal);
};

ATTR_PURE_NDEBUG HOST_DEVICE inline UnitVector
refract_by_normal(float ior, const UnitVector &vec, const UnitVector &normal) {
  float cos_to_normal = -vec->dot(*normal);

  bool exiting_media = cos_to_normal < 0;

  float frac = exiting_media ? ior : 1 / ior;

  float in_sqrt = 1 - frac * frac * (1 - cos_to_normal * cos_to_normal);

  UnitVector effective_normal =
      exiting_media ? UnitVector::new_unchecked(-*normal) : normal;

  if (in_sqrt > 0) {
    return UnitVector::new_normalize(
        Eigen::AngleAxisf(std::acos(std::sqrt(in_sqrt)),
                          vec->cross(*effective_normal).normalized()) *
        -*effective_normal);
  } else {
    return reflect_over_normal(vec, normal);
  }
};
} // namespace bsdf
