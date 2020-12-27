#pragma once

#include "lib/cuda/utils.h"

#include "lib/unit_vector.h"

#include <Eigen/Geometry>

namespace intersect {
struct Ray {
  Eigen::Vector3f origin;
  UnitVector direction;

  HOST_DEVICE inline Ray transform(const Eigen::Affine3f &transform) const {
    return Ray{transform * origin,
               UnitVector::new_normalize(transform.linear() * *direction)};
  }
};
} // namespace intersect
