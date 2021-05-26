#pragma once

#include "generate_data/triangle_subset.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/vector_type.h"

#include <Eigen/Dense>
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

  const auto &l_poly = l.get(tag_v<TriangleSubsetType::Some>);
  const auto &r_poly = r.get(tag_v<TriangleSubsetType::Some>);

  VectorT<TriPolygon> orig_output;
  boost::geometry::intersection(l_poly, r_poly, orig_output);
  // NOTE: see https://github.com/boostorg/geometry/issues/852
  VectorT<TriPolygon> output;
  output.reserve(orig_output.size());
  for (auto &poly : orig_output) {
    VectorT<unsigned> to_retain;
    to_retain.reserve(poly.outer().size() - 1);
    for (unsigned i = 0; i < poly.outer().size() - 1; ++i) {
      unsigned next_i = (i + 1) % poly.outer().size();
      auto p = baryo_to_eigen(poly.outer()[i]);
      auto next_p = baryo_to_eigen(poly.outer()[next_i]);
      if ((next_p - p).norm() > 1e-6) {
        to_retain.push_back(i);
      }
    }
    if (to_retain.size() < 3) {
      continue;
    }
    if (to_retain.size() == poly.outer().size() - 1) {
      output.push_back(std::move(poly));
    } else {
      TriPolygon new_poly;
      for (unsigned retained : to_retain) {
        boost::geometry::append(new_poly, poly.outer()[retained]);
      }
      boost::geometry::append(new_poly, poly.outer()[to_retain[0]]);
      output.push_back(new_poly);
    }
  }
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
