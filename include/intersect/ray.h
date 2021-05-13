#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"

#include "lib/unit_vector.h"

#include <Eigen/Geometry>

namespace intersect {
template <typename T> struct GenRay {
  Eigen::Vector3<T> origin;
  UnitVectorGen<T> direction;

  template <typename Transform>
  ATTR_PURE_NDEBUG HOST_DEVICE inline GenRay
  transform(const Transform &transform) const {
    return {transform * origin,
            UnitVectorGen<T>::new_normalize(transform.linear() * *direction)};
  }
};

using Ray = GenRay<float>;
} // namespace intersect
