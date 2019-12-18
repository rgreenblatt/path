#pragma once

#include "lib/cuda_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ray {
namespace detail {
inline HOST_DEVICE Eigen::Vector2f get_not_axis(const Eigen::Vector3f &vec,
                                                uint8_t axis) {
  return Eigen::Vector2f(vec[(axis + 1) % 3], vec[(axis + 2) % 3]);
}

inline HOST_DEVICE Eigen::Vector2f
get_intersection_point(const Eigen::Vector3f &dir, float value_to_project_to,
                       const Eigen::Vector3f &world_space_eye, uint8_t axis) {
  float dist = (value_to_project_to - world_space_eye[axis]) / dir[axis];

  return (dist * get_not_axis(dir, axis) + get_not_axis(world_space_eye, axis))
      .eval();
}

inline HOST_DEVICE Eigen::Vector3f
initial_world_space_direction(unsigned x, unsigned y, unsigned x_dim,
                              unsigned y_dim,
                              const Eigen::Vector3f &world_space_eye,
                              const Eigen::Affine3f &m_film_to_world) {
  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * static_cast<float>(x)) / static_cast<float>(x_dim) - 1.0f,
      (-2.0f * static_cast<float>(y)) / static_cast<float>(y_dim) + 1.0f,
      -1.0f);
  const auto world_space_film_plane = m_film_to_world * camera_space_film_plane;

  return (world_space_film_plane - world_space_eye).normalized();
}
} // namespace detail
} // namespace ray
