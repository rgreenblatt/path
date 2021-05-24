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
  if (output.empty()) {
    return {tag_v<TriangleSubsetType::None>, {}};
  } else if (output.size() == 1) {
    return {tag_v<TriangleSubsetType::Some>, output[0]};
  } else {
    std::optional<unsigned> actual_poly_idx;
    for (unsigned i = 0; i < output.size(); ++i) {
      if (boost::geometry::area(output[i]) > 1e-13) {
        debug_assert(!actual_poly_idx.has_value());
        actual_poly_idx = i;
      }
    }
    if (!actual_poly_idx.has_value()) {
      // TODO: does this make sense? (maybe doesn't matter too much...)
      return {tag_v<TriangleSubsetType::None>, {}};
    } else {
      return {tag_v<TriangleSubsetType::Some>, output[*actual_poly_idx]};
    }
  }
}
} // namespace generate_data
