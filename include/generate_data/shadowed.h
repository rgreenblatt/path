#pragma once

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/span.h"
#include "lib/vector_type.h"

#include <Eigen/Core>

// All of these functions should work on any set of triangles (doesn't require
// normalization), but onto normals must be pointing the right way.
// However, they may not work on intersecting triangles (untested)
namespace generate_data {
// TODO: expand info
enum class RayItemResultType {
  Ray,
  Intersection,
};

struct PartiallyShadowedInfo {
  struct RayItem {
    using Result =
        TaggedUnion<RayItemResultType, Eigen::Vector2d, Eigen::Vector2d>;
    BaryoPoint baryo_origin;
    BaryoPoint baryo_endpoint;
    Eigen::Vector3d origin;
    Eigen::Vector3d endpoint;
    Result result;
  };

  TriangleSubset partially_shadowed;
  VectorT<RayItem> ray_items;
};

ATTR_PURE_NDEBUG PartiallyShadowedInfo partially_shadowed(
    const intersect::TriangleGen<double> &from,
    const TriangleSubset &from_clipped_region,
    const intersect::TriangleGen<double> &blocker,
    const TriangleSubset &blocker_clipped_region,
    const intersect::TriangleGen<double> &onto, bool flip_onto_normal = false);

ATTR_PURE_NDEBUG TriangleSubset
shadowed_from_point(const Eigen::Vector3d &point,
                    SpanSized<const Eigen::Vector3d> blocker_points,
                    const intersect::TriangleGen<double> &onto);

// TODO: expand info
struct TotallyShadowedInfo {
  TriangleSubset totally_shadowed;
  VectorT<TriangleSubset> from_each_point;
};

ATTR_PURE_NDEBUG TotallyShadowedInfo
totally_shadowed(const intersect::TriangleGen<double> &from,
                 const TriangleSubset &from_clipped_region,
                 const intersect::TriangleGen<double> &blocker,
                 const TriangleSubset &blocker_clipped_region,
                 const intersect::TriangleGen<double> &onto);
} // namespace generate_data
