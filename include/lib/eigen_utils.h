#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <limits>

template <typename Scalar>
ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3<Scalar> max_eigen_vec() {
  return Eigen::Vector3<Scalar>::Constant(std::numeric_limits<Scalar>::max());
}

template <typename Scalar>
ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3<Scalar> min_eigen_vec() {
  return Eigen::Vector3<Scalar>::Constant(
      std::numeric_limits<Scalar>::lowest());
}
