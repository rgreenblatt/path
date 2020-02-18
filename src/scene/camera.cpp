#include "scene/camera.h"

namespace scene {
Eigen::Affine3f get_camera_transform(const Eigen::Vector3f &look,
                                     const Eigen::Vector3f &up,
                                     const Eigen::Vector3f &pos,
                                     float height_angle,
                                     float width_height_ratio) {
  float theta_h = height_angle;
  float theta_w = std::atan(width_height_ratio * std::tan(theta_h / 2)) * 2;

  auto mat = look_at(look, up);

  auto f = (mat * Eigen::Translation3f(-pos)).inverse();

  Eigen::Vector3f scaling_vec(1.0f / (std::tan(theta_w / 2)),
                              1.0f / (std::tan(theta_h / 2)), 1.0f);

  Eigen::Affine3f scaling(Eigen::Scaling(scaling_vec));

  auto film_to_world = f * scaling.inverse();

  return film_to_world;
}
} // namespace scene
