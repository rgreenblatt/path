#pragma once

#include "lib/attribute.h"
#include "lib/tagged_union.h"
#include "meta/all_values/impl/enum.h"

#include <Eigen/Core>
#include <boost/geometry/geometries/multi_polygon.hpp>
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
using TriMultiPolygon = boost::geometry::model::multi_polygon<TriPolygon>;

ATTR_PURE_NDEBUG inline BaryoPoint eigen_to_baryo(const Eigen::Vector2d &p) {
  return {p.x(), p.y()};
}

ATTR_PURE_NDEBUG inline Eigen::Vector2d baryo_to_eigen(const BaryoPoint &p) {
  return {p.x(), p.y()};
}

// TODO: consider if it would make sense to improve
// perf by using fixed size values in some cases...
using TriangleSubset =
    TaggedUnion<TriangleSubsetType, std::tuple<>, std::tuple<>, TriPolygon>;

using TriangleMultiSubset = TaggedUnion<TriangleSubsetType, std::tuple<>,
                                        std::tuple<>, TriMultiPolygon>;

const static TriPolygon full_triangle = {
    {{0., 0.}, {0., 1.}, {1., 0.}, {0., 0.}}};
} // namespace generate_data
