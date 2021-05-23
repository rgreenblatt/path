#pragma once

#include "generate_data/triangle_subset.h"
#include "lib/assert.h"
#include "lib/attribute.h"

#include <boost/geometry.hpp>

namespace generate_data {
ATTR_PURE_NDEBUG inline TriangleSubset
triangle_subset_intersection(const TriangleSubset &l, const TriangleSubset &r) {
  if (l.type() == TriangleSubsetType::None) {
    return l;
  }
  if (r.type() == TriangleSubsetType::None) {
    return r;
  }
  if (l.type() == TriangleSubsetType::All) {
    return r;
  }
  if (r.type() == TriangleSubsetType::All) {
    return l;
  }

  auto l_poly = l.get(tag_v<TriangleSubsetType::Some>);
  auto r_poly = r.get(tag_v<TriangleSubsetType::Some>);

  std::vector<TriPolygon> output;
  boost::geometry::intersection(l_poly, r_poly, output);
  always_assert(output.size() <= 1);

  if (output.empty()) {
    return {tag_v<TriangleSubsetType::None>, {}};
  }

  return {tag_v<TriangleSubsetType::Some>, output[0]};
}
} // namespace generate_data
