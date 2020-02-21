#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

inline HOST_DEVICE Eigen::AngleAxisf
find_rotate_vector_to_vector(const Eigen::Vector3f &inp,
                             const Eigen::Vector3f &target) {
  auto inp_normalized = inp.normalized().eval();
  auto target_normalized = target.normalized().eval();
  float angle_between = std::acos(inp_normalized.dot(target_normalized));
  auto orthagonal = (inp_normalized.cross(target_normalized)).normalized();

  return Eigen::AngleAxisf(angle_between, orthagonal);
}

inline HOST_DEVICE Eigen::Vector3f find_relative_vec(const Eigen::Vector3f &vec,
                                                     float phi, float theta) {
  float sin_theta = std::sin(theta);
  return (find_rotate_vector_to_vector(vec, {0, 0, 1}).inverse() *
          Eigen::Vector3f(sin_theta * std::cos(phi), sin_theta * std::sin(phi),
                          std::cos(theta)))
      .normalized();
}
