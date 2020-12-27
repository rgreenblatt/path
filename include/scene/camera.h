#pragma once

#include "lib/unit_vector.h"

#include <Eigen/Geometry>

namespace scene {
Eigen::Affine3f get_camera_transform(const UnitVector &look,
                                     const UnitVector &up,
                                     const Eigen::Vector3f &pos,
                                     float height_angle,
                                     float width_height_ratio);
} // namespace scene
