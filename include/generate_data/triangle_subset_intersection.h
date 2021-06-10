#pragma once

#include "generate_data/triangle_subset.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/vector_type.h"

#include "dbg.h"

#include <boost/geometry.hpp>

namespace generate_data {
namespace detail {
template <typename T, typename F>
ATTR_PURE_NDEBUG inline T
triangle_subset_intersection_gen(const T &l, const T &r, const F &other_case) {
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

  return other_case(l.get(tag_v<TriangleSubsetType::Some>),
                    r.get(tag_v<TriangleSubsetType::Some>));
}
} // namespace detail

ATTR_PURE_NDEBUG inline TriangleSubset
triangle_subset_intersection(const TriangleSubset &l, const TriangleSubset &r) {
  return detail::triangle_subset_intersection_gen(
      l, r,
      [&](const TriPolygon &l_poly,
          const TriPolygon &r_poly) -> TriangleSubset {
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
            // TODO: more robust approach would just take max?
            if (boost::geometry::area(output[i]) > 1e-10) {
              if (actual_poly_idx.has_value()) {
                for (const auto &out : output) {
                  dbg(boost::geometry::area(out));
                }
              }
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
      });
}

ATTR_PURE_NDEBUG inline TriangleMultiSubset
triangle_multi_subset_intersection(const TriangleMultiSubset &l,
                                   const TriangleMultiSubset &r) {
  return detail::triangle_subset_intersection_gen(
      l, r,
      [&](const TriMultiPolygon &l_poly,
          const TriMultiPolygon &r_poly) -> TriangleMultiSubset {
        TriMultiPolygon output;
        boost::geometry::intersection(l_poly, r_poly, output);

        // TODO: any need to None check this?
        // like case when area is small
        return {tag_v<TriangleSubsetType::Some>, output};
      });
}
} // namespace generate_data
