#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "ray/detail/block_data.h"

#include <Eigen/Geometry>

namespace ray {
namespace detail {
inline HOST_DEVICE void initial_world_space_directions_impl(
    unsigned block_index, unsigned thread_index, const BlockData &block_data,
    const Eigen::Vector3f &world_space_eye,
    const Eigen::Affine3f &m_film_to_world,
    Span<Eigen::Vector3f> world_space_directions) {
  auto [x, y, index, outside_bounds] =
      block_data.getIndexes(block_index, thread_index);

  if (outside_bounds) {
    return;
  }

  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * static_cast<float>(x)) / static_cast<float>(block_data.x_dim) -
          1.0f,
      (-2.0f * static_cast<float>(y)) / static_cast<float>(block_data.y_dim) +
          1.0f,
      -1.0f);
  const auto world_space_film_plane = m_film_to_world * camera_space_film_plane;

  world_space_directions[index] =
      (world_space_film_plane - world_space_eye).normalized();
}
} // namespace detail
} // namespace ray
