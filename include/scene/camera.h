#pragma once

#include <Eigen/Geometry>
#include <thrust/optional.h>

namespace scene {
std::tuple<Eigen::Affine3f, Eigen::Projective3f>
get_camera_transform(const Eigen::Vector3f &look, const Eigen::Vector3f &up,
                     const Eigen::Vector3f &pos, float height_angle,
                     float width, float height, float far = 30.0f,
                     thrust::optional<Eigen::Vector3f> scale = thrust::nullopt);

inline Eigen::Projective3f get_unhinging(float far) {
  Eigen::Projective3f unhinging = Eigen::Projective3f::Identity();
  float c = -1.0f / far;
  unhinging(2, 2) = -1.0f / (c + 1);
  unhinging(2, 3) = c / (c + 1);
  unhinging(3, 2) = -1;
  unhinging(3, 3) = 0;

  return unhinging;
}

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
