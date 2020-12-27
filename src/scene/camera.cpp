#include "scene/camera.h"

namespace scene {
Eigen::Affine3f get_camera_transform(const UnitVector &look,
                                     const UnitVector &up,
                                     const Eigen::Vector3f &pos,
                                     float height_angle,
                                     float width_height_ratio) {

  const Eigen::Vector3f &f = *look;
  Eigen::Vector3f s = f.cross(*up);
  Eigen::Vector3f u = s.cross(f);

  Eigen::Matrix4f view_mat;
  view_mat << s.x(), s.y(), s.z(), -s.dot(pos), u.x(), u.y(), u.z(),
      -u.dot(pos), -f.x(), -f.y(), -f.z(), f.dot(pos), 0, 0, 0, 1;

  float height_angle_rads = M_PI * height_angle / 360.f; // We need half the
                                                         // angle
  float tan_theta_h = tan(height_angle_rads);
  float tan_theta_w = width_height_ratio * tan_theta_h;

  Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
  scale(0, 0) = 1 / tan_theta_w;
  scale(1, 1) = 1 / tan_theta_h;

  return Eigen::Affine3f((scale * view_mat).inverse());
}
} // namespace scene
