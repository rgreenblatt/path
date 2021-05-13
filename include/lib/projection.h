#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/unit_vector.h"

#include <Eigen/Geometry>

template <typename T>
ATTR_PURE_NDEBUG inline HOST_DEVICE Eigen::AngleAxis<T>
find_rotate_vector_to_vector(const UnitVectorGen<T> &inp,
                             const UnitVectorGen<T> &target) {
  // SPEED: better approach
  // SPEED: cache/don't recompute cos etc
  T angle_between = std::acos(inp->dot(*target));
  auto orthagonal = (inp->cross(*target)).normalized();

  return Eigen::AngleAxis<T>(angle_between, orthagonal);
}

template <typename T>
ATTR_PURE_NDEBUG inline HOST_DEVICE UnitVectorGen<T>
find_relative_vec(const UnitVectorGen<T> &vec, T phi, T theta) {
  // SPEED: better approach
  T sin_theta = std::sin(theta);
  return UnitVectorGen<T>::new_normalize(
      find_rotate_vector_to_vector(
          vec, UnitVectorGen<T>::new_unchecked({0., 0., 1.}))
          .inverse() *
      Eigen::Vector3<T>(sin_theta * std::cos(phi), sin_theta * std::sin(phi),
                        std::cos(theta)));
}
