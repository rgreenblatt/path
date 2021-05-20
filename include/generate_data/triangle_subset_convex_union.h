#pragma once

#include "generate_data/triangle_subset.h"
#include "lib/assert.h"
#include "lib/attribute.h"

#include <boost/geometry.hpp>

// I can't seem to figure out the minimal headers...
// #include <boost/geometry/algorithms/append.hpp>
// #include <boost/geometry/algorithms/convex_hull.hpp>
// #include <boost/geometry/geometries/multi_point.hpp>

// TODO: consider moving impl to cpp file
namespace generate_data {
template <typename Range>
ATTR_PURE_NDEBUG inline TriangleSubset
triangle_subset_convex_union(const Range &subsets) {
  bool all_none = true;
  for (const auto &sub : subsets) {
    if (sub.type() == TriangleSubsetType::All) {
      return sub;
    }
    all_none = all_none && sub.type() == TriangleSubsetType::None;
  }
  if (all_none) {
    return {tag_v<TriangleSubsetType::None>, {}};
  }

  boost::geometry::model::multi_point<BaryoPoint> points;
  for (const auto &sub : subsets) {
    if (sub.type() == TriangleSubsetType::Some) {
      auto poly = sub.get(tag_v<TriangleSubsetType::Some>);
      for (const auto &p : poly.outer()) {
        boost::geometry::append(points, p);
      }
    }
  }
  debug_assert(!points.empty());

  TriPolygon out;
  boost::geometry::convex_hull(points, out);

  return {tag_v<TriangleSubsetType::Some>, out};
}
} // namespace generate_data
