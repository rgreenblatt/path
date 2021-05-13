#pragma once

#include "lib/attribute.h"
#include "lib/unit_vector.h"

#include <Eigen/Geometry>

namespace scene {
ATTR_PURE_NDEBUG Eigen::Affine3f
get_camera_transform(const UnitVector &look, const UnitVector &up,
                     const Eigen::Vector3f &pos, float height_angle_deg,
                     float width_height_ratio);
} // namespace scene
