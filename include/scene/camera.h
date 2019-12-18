#pragma once

#include <Eigen/Geometry>
#include <thrust/optional.h>

namespace scene {
std::tuple<Eigen::Affine3f, Eigen::Projective3f>
get_camera_transform(const Eigen::Vector3f &look, const Eigen::Vector3f &up,
                     const Eigen::Vector3f &pos, float height_angle,
                     float width, float height, float far = 30.0f,
                     thrust::optional<Eigen::Vector3f> scale = thrust::nullopt);

} // namespace scene
