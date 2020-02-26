#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

namespace intersect {
struct Ray {
  Eigen::Vector3f origin;
  Eigen::Vector3f direction;

  HOST_DEVICE inline Ray transform(const Eigen::Affine3f &transform) const;
};
} // namespace intersect
