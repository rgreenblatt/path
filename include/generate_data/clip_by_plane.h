#pragma once

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"

#include <Eigen/Core>

namespace generate_data {
ATTR_PURE_NDEBUG TriangleSubset
clip_by_plane(const Eigen::Vector3d &normal, const Eigen::Vector3d &point,
              const intersect::TriangleGen<double> &tri);
} // namespace generate_data
