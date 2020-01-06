#include "scene/camera.h"

namespace scene {
std::tuple<Eigen::Affine3f, Eigen::Projective3f>
get_camera_transform(const Eigen::Vector3f &look, const Eigen::Vector3f &up,
                     const Eigen::Vector3f &pos, float height_angle,
                     float width, float height, float far,
                     thrust::optional<Eigen::Vector3f> scale) {
  float theta_h = height_angle;
  float theta_w = std::atan(width * std::tan(theta_h / 2) / height) * 2;

  auto mat = look_at(look, up);

  auto f = (mat * Eigen::Translation3f(-pos)).inverse();

  Eigen::Vector3f scaling_vec =
      scale.has_value()
          ? *scale
          : Eigen::Vector3f(1.0f / (std::tan(theta_w / 2) * far),
                            1.0f / (std::tan(theta_h / 2) * far), 1.0f / far);

  Eigen::Affine3f scaling =
      Eigen::Scaling(scaling_vec) * Eigen::Affine3f::Identity();

  auto film_to_world = f * scaling.inverse();

  Eigen::Projective3f world_to_film =
      get_unhinging(far) * static_cast<Eigen::Projective3f>(
                               scaling * mat * Eigen::Translation3f(-pos));

  return std::make_tuple(film_to_world, world_to_film);
}
} // namespace scene
