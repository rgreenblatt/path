#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <limits>

ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3f max_eigen_vec() {
  return Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
}

ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3f min_eigen_vec() {
  return Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
}
