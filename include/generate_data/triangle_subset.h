#pragma once

#include "lib/tagged_union.h"
#include "meta/all_values/impl/enum.h"

#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace generate_data {
enum class TriangleSubsetType {
  All,
  None,
  Some,
};

using BaryoPoint = boost::geometry::model::d2::point_xy<double>;
using TriPolygon = boost::geometry::model::polygon<BaryoPoint>;

// TODO: consider if it would make sense to use improve
// perf by using fixed size values in some cases...
using TriangleSubset =
    TaggedUnion<TriangleSubsetType, std::tuple<>, std::tuple<>, TriPolygon>;
} // namespace generate_data
