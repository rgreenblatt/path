#include "scene/camera.h"

#include "ray/kdtree.h"
#include "ray/projection.h"
#include <dbg.h>

namespace scene {
std::tuple<Eigen::Affine3f, Eigen::Affine3f, Eigen::Projective3f>
get_camera_transform(const Eigen::Vector3f &look, const Eigen::Vector3f &up,
                     const Eigen::Vector3f &pos, float height_angle,
                     float width, float height, float far,
                     thrust::optional<Eigen::Vector3f> scale) {
  auto w = -look.normalized().eval();
  auto normalized_up = up.normalized().eval();
  auto v = (normalized_up - normalized_up.dot(w) * w).normalized().eval();
  auto u = v.cross(w);

  Eigen::Matrix3f mat;

  mat.row(0) = u;
  mat.row(1) = v;
  mat.row(2) = w;

  float theta_h = height_angle;
  float theta_w = std::atan(width * std::tan(theta_h / 2) / height) * 2;

  auto f = (mat * Eigen::Translation3f(-pos)).inverse();

  Eigen::Vector3f scaling_vec =
      scale.has_value()
          ? *scale
          : Eigen::Vector3f(1.0f / (std::tan(theta_w / 2) * far),
                            1.0f / (std::tan(theta_h / 2) * far), 1.0f / far);

  Eigen::Affine3f scaling =
      Eigen::Scaling(scaling_vec) * Eigen::Affine3f::Identity();

  auto film_to_world = f * scaling.inverse();

  Eigen::Projective3f unhinging = Eigen::Projective3f::Identity();
  float c = -1.0f / far;
  unhinging(2, 2) = -1.0f / (c + 1);
  unhinging(2, 3) = c / (c + 1);
  unhinging(3, 2) = -1;
  unhinging(3, 3) = 0;

  Eigen::Affine3f world_to_film = scaling * mat * Eigen::Translation3f(-pos);

  return std::make_tuple(film_to_world, world_to_film, unhinging);
}
} // namespace scene
