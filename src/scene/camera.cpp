#include "scene/camera.h"

namespace scene {
Eigen::Affine3f get_camera_transform(const Eigen::Vector3f &look,
                                            const Eigen::Vector3f &up,
                                            const Eigen::Vector3f &pos,
                                            float height_angle, unsigned width,
                                            unsigned height) {
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

  return f *
         Eigen::Scaling(Eigen::Vector3f(1.0f / (std::tan(theta_w / 2) * far),
                                        1.0f / (std::tan(theta_h / 2) * far),
                                        1.0f / far))
             .inverse();
}

} // namespace scene
