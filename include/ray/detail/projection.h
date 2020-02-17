#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

namespace ray {
namespace detail {
inline HOST_DEVICE Eigen::Vector3f
apply_projective_point(const Eigen::Vector3f &point,
                       const Eigen::Projective3f &projection) {
  Eigen::Vector4f homog;
  homog.template head<3>() = point;
  homog[3] = 1.0f;
  auto out = (projection * homog).eval();

  return (out.head<3>() / out[3]).eval();
}

// this could probably be faster
inline HOST_DEVICE Eigen::Vector3f
apply_projective_vec(const Eigen::Vector3f &vec,
                     const Eigen::Projective3f &projection) {
  return apply_projective_point(vec, projection) -
         projection.translation() / projection(3, 3);
}

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
