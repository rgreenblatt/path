#pragma once

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/vector_type.h"

namespace generate_data {
struct PointsWithBaryo {
  VectorT<Eigen::Vector3d> points;
  VectorT<BaryoPoint> baryo;
};

template <bool has_baryo>
ATTR_PURE_NDEBUG inline std::conditional_t<has_baryo, PointsWithBaryo,
                                           VectorT<Eigen::Vector3d>>
get_points_from_subset_with_baryo_impl(
    const intersect::TriangleGen<double> &tri, const TriangleSubset &subset) {
  const auto out = subset.visit_tagged(
      [&](auto tag,
          const auto &poly) -> std::conditional_t<has_baryo, PointsWithBaryo,
                                                  VectorT<Eigen::Vector3d>> {
        if constexpr (tag == TriangleSubsetType::None) {
          return {};
        } else if constexpr (tag == TriangleSubsetType::All) {
          VectorT<Eigen::Vector3d> points{
              tri.vertices[0],
              tri.vertices[2],
              tri.vertices[1],
          };
          if constexpr (has_baryo) {
            return {.points = points,
                    .baryo = {
                        {0., 0.},
                        {0., 1.},
                        {1., 0.},
                    }};
          } else {
            return points;
          }
        } else {
          static_assert(tag == TriangleSubsetType::Some);

          debug_assert(poly.outer().size() >= 4);
          // -1 because poly loops back to original point
          VectorT<Eigen::Vector3d> points(poly.outer().size() - 1);
          std::transform(poly.outer().begin(), poly.outer().end() - 1,
                         points.begin(), [&](const BaryoPoint &point) {
                           return tri.baryo_to_point({point.x(), point.y()});
                         });
          if constexpr (has_baryo) {
            return {.points = points,
                    .baryo = {poly.outer().begin(), poly.outer().end() - 1}};
          } else {
            return points;
          }
        }
      });
  if constexpr (has_baryo) {
    debug_assert(out.points.size() == out.baryo.size());
  }

  return out;
}

ATTR_PURE_NDEBUG inline VectorT<Eigen::Vector3d>
get_points_from_subset(const intersect::TriangleGen<double> &tri,
                       const TriangleSubset &subset) {
  return get_points_from_subset_with_baryo_impl<false>(tri, subset);
}

ATTR_PURE_NDEBUG inline PointsWithBaryo
get_points_from_subset_with_baryo(const intersect::TriangleGen<double> &tri,
                                  const TriangleSubset &subset) {
  return get_points_from_subset_with_baryo_impl<true>(tri, subset);
}
} // namespace generate_data
