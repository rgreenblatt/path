#pragma once

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"

#include <Eigen/Core>

namespace generate_data {
ATTR_PURE_NDEBUG TriangleSubset
clip_by_plane(const Eigen::Vector3d &normal, double plane_threshold,
              const intersect::TriangleGen<double> &tri);

ATTR_PURE_NDEBUG inline TriangleSubset
clip_by_plane_point(const Eigen::Vector3d &normal, const Eigen::Vector3d &point,
                    const intersect::TriangleGen<double> &tri) {
  return clip_by_plane(normal, normal.dot(point), tri);
}
} // namespace generate_data
