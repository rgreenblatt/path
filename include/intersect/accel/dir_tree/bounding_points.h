#pragma once

#include <Eigen/Geometry>

#include <array>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
using BoundingPoints = std::array<Eigen::Vector3f, 8>;

inline BoundingPoints get_bounding(const Eigen::Affine3f &transform_v) {
  auto trans = [&](const Eigen::Vector3f &point) {
    return transform_v * point;
  };

  return {
      trans({0.5f, 0.5f, 0.5f}),   trans({-0.5f, 0.5f, 0.5f}),
      trans({0.5f, -0.5f, 0.5f}),  trans({0.5f, 0.5f, -0.5f}),
      trans({-0.5f, -0.5f, 0.5f}), trans({0.5f, -0.5f, -0.5f}),
      trans({-0.5f, 0.5f, -0.5f}), trans({-0.5f, -0.5f, -0.5f}),
  };
}

} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
