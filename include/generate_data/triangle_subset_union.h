#pragma once

#include "generate_data/triangle_subset.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/vector_type.h"

#include <boost/geometry.hpp>

namespace generate_data {
ATTR_PURE_NDEBUG inline TriangleMultiSubset
triangle_subset_union(const TriangleMultiSubset &l,
                      const TriangleMultiSubset &r) {
  if (l.type() == TriangleSubsetType::None) {
    return r;
  }
  if (r.type() == TriangleSubsetType::None) {
    return l;
  }
  if (l.type() == TriangleSubsetType::All) {
    return l;
  }
  if (r.type() == TriangleSubsetType::All) {
    return r;
  }

  const auto &l_poly = l.get(tag_v<TriangleSubsetType::Some>);
  const auto &r_poly = r.get(tag_v<TriangleSubsetType::Some>);

  TriMultiPolygon output;
  boost::geometry::union_(l_poly, r_poly, output);

  return {tag_v<TriangleSubsetType::Some>, output};
}
} // namespace generate_data
