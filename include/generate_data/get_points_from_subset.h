#pragma once

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/vector_type.h"

namespace generate_data {
ATTR_PURE_NDEBUG inline VectorT<Eigen::Vector3d>
get_points_from_subset(const intersect::TriangleGen<double> &tri,
                       const TriangleSubset &subset) {
  return subset.visit_tagged(
      [&](auto tag, const auto &poly) -> VectorT<Eigen::Vector3d> {
        if constexpr (tag == TriangleSubsetType::None) {
          return {};
        } else if constexpr (tag == TriangleSubsetType::All) {
          return {
              tri.vertices[0],
              tri.vertices[1],
              tri.vertices[2],
          };
        } else {
          static_assert(tag == TriangleSubsetType::Some);

          debug_assert(poly.outer().size() >= 4);
          // -1 because poly loops back to original point
          std::vector<Eigen::Vector3d> out(poly.outer().size() - 1);
          std::transform(poly.outer().begin(), poly.outer().end() - 1,
                         out.begin(), [&](const BaryoPoint &point) {
                           return tri.value_from_baryo({point.x(), point.y()});
                         });
          return out;
        }
      });
}
} // namespace generate_data
