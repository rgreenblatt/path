#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

namespace ray {
namespace detail {
inline HOST_DEVICE Eigen::Affine3f
find_rotate_vector_to_vector(const Eigen::Vector3f &inp,
                             const Eigen::Vector3f &target) {
  auto inp_normalized = inp.normalized().eval();
  auto target_normalized = target.normalized().eval();
  float angle_between = std::acos(inp_normalized.dot(target_normalized));
  auto orthagonal = (inp_normalized.cross(target_normalized)).normalized();

  return Eigen::Affine3f(Eigen::AngleAxisf(angle_between, orthagonal));
}
} // namespace detail
} // namespace ray
