#pragma once

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/vector_type.h"

#include <Eigen/Core>

namespace generate_data {
ATTR_PURE_NDEBUG TriangleSubset shadowed_from_point(
    const Eigen::Vector3d &point, const intersect::TriangleGen<double> &blocker,
    const intersect::TriangleGen<double> &onto);

struct ShadowedInfo {
  TriangleSubset some_blocking;
  TriangleSubset totally_blocked;
  VectorT<TriangleSubset> from_each_point;
};

ATTR_PURE_NDEBUG ShadowedInfo
shadowed(const VectorT<Eigen::Vector3d> &from_points,
         const intersect::TriangleGen<double> &blocker,
         const intersect::TriangleGen<double> &onto);
} // namespace generate_data
