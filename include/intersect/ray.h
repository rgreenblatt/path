#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

namespace intersect {
struct Ray {
  Eigen::Vector3f origin;
  Eigen::Vector3f direction; // must be normalized

  HOST_DEVICE inline Ray transform(const Eigen::Affine3f &transform) const {
    // TODO why where they using inverse().transpose()?
    return Ray{transform * origin, transform.linear() * direction};
  }
};
} // namespace intersect
