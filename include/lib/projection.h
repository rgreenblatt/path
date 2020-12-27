#pragma once

#include "lib/cuda/utils.h"
#include "lib/unit_vector.h"

#include <Eigen/Geometry>

inline HOST_DEVICE Eigen::AngleAxisf
find_rotate_vector_to_vector(const UnitVector &inp, const UnitVector &target) {
  // SPEED: better approach
  // SPEED: cache/don't recompute cos etc
  float angle_between = std::acos(inp->dot(*target));
  auto orthagonal = (inp->cross(*target)).normalized();

  return Eigen::AngleAxisf(angle_between, orthagonal);
}

inline HOST_DEVICE UnitVector find_relative_vec(const UnitVector &vec,
                                                float phi, float theta) {
  // SPEED: better approach
  float sin_theta = std::sin(theta);
  return UnitVector::new_normalize(
      find_rotate_vector_to_vector(vec,
                                   UnitVector::new_unchecked({0.f, 0.f, 1.f}))
          .inverse() *
      Eigen::Vector3f(sin_theta * std::cos(phi), sin_theta * std::sin(phi),
                      std::cos(theta)));
}
