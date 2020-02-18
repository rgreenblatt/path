#pragma once

#include <Eigen/Geometry>
#include <thrust/optional.h>

namespace scene {
Eigen::Affine3f get_camera_transform(const Eigen::Vector3f &look,
                                     const Eigen::Vector3f &up,
                                     const Eigen::Vector3f &pos,
                                     float height_angle,
                                     float width_height_ratio);

inline Eigen::Matrix3f look_at(const Eigen::Vector3f &look,
                               const Eigen::Vector3f &up) {
  auto w = -look.normalized().eval();
  auto normalized_up = up.normalized().eval();
  auto v = (normalized_up - normalized_up.dot(w) * w).normalized().eval();
  auto u = v.cross(w);

  Eigen::Matrix3f mat;

  mat.row(0) = u;
  mat.row(1) = v;
  mat.row(2) = w;

  return mat;
}
} // namespace scene
