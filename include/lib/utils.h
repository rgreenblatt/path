#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <limits>

constexpr unsigned ceil_divide(unsigned l, unsigned r) {
  return (l + r - 1) / r;
};

constexpr unsigned make_divisable(unsigned v, unsigned divisable_by) {
  return ceil_divide(v, divisable_by) * divisable_by;
}

// Testing:
static_assert(make_divisable(3, 1) == 3);
static_assert(make_divisable(4, 1) == 4);
static_assert(make_divisable(4, 2) == 4);
static_assert(make_divisable(4, 3) == 6);
static_assert(make_divisable(3, 3) == 3);
static_assert(make_divisable(100, 3) == 102);

HOST_DEVICE inline Eigen::Vector3f max_eigen_vec() {
  return Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
}

HOST_DEVICE inline Eigen::Vector3f min_eigen_vec() {
  return Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
}
