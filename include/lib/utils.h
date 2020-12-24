#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <limits>

constexpr unsigned ceil_divide(unsigned l, unsigned r) {
  return (l + r - 1) / r;
};

HOST_DEVICE inline Eigen::Vector3f max_eigen_vec() {
  return Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
}

HOST_DEVICE inline Eigen::Vector3f min_eigen_vec() {
  return Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
}
