#include "scene/camera.h"

#include <dbg.h>

namespace scene {
std::tuple<Eigen::Affine3f, Eigen::Projective3f>
get_camera_transform(const Eigen::Vector3f &look, const Eigen::Vector3f &up,
                     const Eigen::Vector3f &pos, float height_angle,
                     unsigned width, unsigned height) {
#if 1
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
  float far = 30.0f;

  auto f = (mat * Eigen::Translation3f(-pos)).inverse();

  Eigen::Vector3f scaling_vec(1.0f / (std::tan(theta_w / 2) * far),
                              1.0f / (std::tan(theta_h / 2) * far), 1.0f / far);

  auto film_to_world = f * Eigen::Scaling(scaling_vec).inverse();

  Eigen::Matrix4f unhinging = Eigen::Matrix4f::Identity();
  float c = -1.0f / 30.0f;
  unhinging(2, 2) = -1.0f / (c + 1);
  unhinging(2, 3) = c / (c + 1);
  unhinging(3, 2) = -1;
  unhinging(3, 3) = 0;

  Eigen::Projective3f world_to_film =
      unhinging *
      static_cast<Eigen::Projective3f>(Eigen::Scaling(scaling_vec) * mat *
                                       Eigen::Translation3f(-pos));
  return std::make_tuple(film_to_world, world_to_film);
#else
  auto w = -look.normalized().eval();
  auto v = (up - up.dot(w) * w).normalized().eval();
  auto u = v.cross(w).normalized();

  Eigen::Matrix3f mat;

  mat.row(0) = u;
  mat.row(1) = v;
  mat.row(2) = w;

  float theta_h = height_angle;
  float theta_w = std::atan(width * std::tan(theta_h / 2) / height) * 2;
  float far = 30.0f;

  auto f = (mat * Eigen::Translation3f(-pos)).inverse();

  return std::make_tuple(
      f * Eigen::Scaling(Eigen::Vector3f(1.0f / (std::tan(theta_w / 2) * far),
                                         1.0f / (std::tan(theta_h / 2) * far),
                                         1.0f / far))
              .inverse(),
      Eigen::Projective3f());
#endif
}

} // namespace scene
