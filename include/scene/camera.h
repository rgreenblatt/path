#pragma once

#include <Eigen/Geometry>

namespace scene {
std::tuple<Eigen::Affine3f, Eigen::Projective3f>
get_camera_transform(const Eigen::Vector3f &look, const Eigen::Vector3f &up,
                     const Eigen::Vector3f &pos, float height_angle,
                     unsigned width, unsigned height);

} // namespace scene
