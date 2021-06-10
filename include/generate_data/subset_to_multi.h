#pragma once

#include "generate_data/triangle_subset.h"

#include <boost/geometry.hpp>

namespace generate_data {
TriangleMultiSubset subset_to_multi(const TriangleSubset &subset) {
  return subset.visit_tagged(
      [](auto tag, const auto &val) -> TriangleMultiSubset {
        if constexpr (tag == TriangleSubsetType::Some) {
          return {tag, {val}};
        } else {
          return {tag, {}};
        }
      });
}
} // namespace generate_data
