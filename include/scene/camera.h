#pragma once

#include <Eigen/Geometry>

namespace scene {
Eigen::Affine3f get_camera_transform(const Eigen::Vector3f &look,
                                     const Eigen::Vector3f &up,
                                     const Eigen::Vector3f &pos,
                                     float height_angle,
                                     float width_height_ratio);
} // namespace scene
