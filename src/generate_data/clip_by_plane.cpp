#include "generate_data/clip_by_plane.h"

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/array_vec.h"

#include <boost/geometry.hpp>

#include "dbg.h"

namespace generate_data {
ATTR_PURE_NDEBUG TriangleSubset
clip_by_plane(const Eigen::Vector3d &normal, double plane_threshold,
              const intersect::TriangleGen<double> &tri, bool debug) {
  ArrayVec<unsigned, 3> included;
  ArrayVec<double, 3> precomputed_vals;
  ArrayVec<unsigned, 3> excluded_or_on_edge;
  unsigned excluded_count = 0;
  if (debug) {
    dbg(normal);
    dbg(plane_threshold);
  }

  std::array<double, 3> vals_all;

  for (unsigned i = 0; i < 3; ++i) {
    double dotted = normal.dot(tri.vertices[i]);
    double value = plane_threshold - dotted;
    if (debug) {
      dbg(tri.vertices[i]);
      dbg(value);
    }
    vals_all[i] = value;
    if (value > -1e-6) {
      excluded_or_on_edge.push_back(i);
      if (value > 1e-6) {
        ++excluded_count;
      }
    } else {
      precomputed_vals.push_back(value);
      included.push_back(i);
    }
  }

  debug_assert(excluded_or_on_edge.size() + included.size() == 3);

  // if all are on edge, we include
  if (excluded_count == 0) {
    return {tag_v<TriangleSubsetType::All>, {}};
  }
  if (included.empty()) {
    return {tag_v<TriangleSubsetType::None>, {}};
  }

  if (debug) {
    dbg(excluded_count);
    dbg(unsigned(included.size()));
  }

  auto make_point = [&](double value, unsigned idx) -> Eigen::Vector2d {
    if (idx == 2) {
      return {0., value};
    } else {
      debug_assert(idx == 1);
      return {value, 0.};
    }
  };

  auto edge_points = [&](unsigned included_idx, unsigned end_idx) {
    unsigned origin_idx = included[included_idx];
    Eigen::Vector3d origin = tri.vertices[origin_idx];
    Eigen::Vector3d ray = tri.vertices[end_idx] - origin;

    // NOTE: ray.dot(normal) can't be zero (outside of some serious floating
    // point issues) because that would imply that either all points are
    // included or all are excluded.
    double prop = precomputed_vals[included_idx] / ray.dot(normal);
    Eigen::Vector3d endpoint = prop * ray + origin;

    // debug_assert(-1e-2 <= prop);
    // debug_assert(1. + 1e-2 >= prop);
    prop = std::clamp(prop, 0., 1.);
    debug_assert(std::abs(endpoint.dot(normal) - plane_threshold) < 1e-6);

    if (origin_idx == 0) {
      return make_point(prop, end_idx);
    } else if (end_idx == 0) {
      return make_point(1 - prop, origin_idx);
    } else {
      auto baryocentric_vals = tri.interpolation_values(endpoint);
      Eigen::Vector2d ret{baryocentric_vals[1], baryocentric_vals[2]};
      [[maybe_unused]] auto baryo_point =
          tri.baryo_to_point({ret.x(), ret.y()});
      debug_assert((endpoint - baryo_point).norm() < 1e-6);

      return ret;
    }
  };

  ArrayVec<Eigen::Vector2d, 4> points;

  auto add_point = [&](const Eigen::Vector2d &p) { points.push_back(p); };
  if (included.size() == 2) {
    add_point(edge_points(0, excluded_or_on_edge[0]));
    add_point(edge_points(1, excluded_or_on_edge[0]));
  } else {
    debug_assert(included.size() == 1);
    add_point(edge_points(0, excluded_or_on_edge[0]));
    add_point(edge_points(0, excluded_or_on_edge[1]));
  }

  auto get_point_in_baryo = [&](unsigned idx) -> Eigen::Vector2d {
    if (idx == 0) {
      return {0., 0.};
    } else {
      return make_point(1., idx);
    }
  };

  for (unsigned i : included) {
    add_point(get_point_in_baryo(i));
  }

  Eigen::Vector2d centroid{0, 0};
  for (const auto &p : points) {
    centroid += p;
  }
  centroid /= points.size();

  ArrayVec<std::tuple<BaryoPoint, double>, 4> points_angle;

  for (const auto &p : points) {
    Eigen::Vector2d vec = p - centroid;
    points_angle.push_back({{p.x(), p.y()}, std::atan2(vec.y(), vec.x())});
  }

  // sort by angle to ensure valid poly
  std::sort(points_angle.begin(), points_angle.end(),
            [](const auto &l, const auto &r) {
              return std::get<1>(l) > std::get<1>(r);
            });

  TriPolygon poly;
  for (const auto &p : points_angle) {
    boost::geometry::append(poly, std::get<0>(p));
  }
  boost::geometry::append(poly, std::get<0>(points_angle[0]));

  double area = boost::geometry::area(poly);
  if (debug) {
    dbg(area);
  }
  if (area <= 1e-12) {
    return {tag_v<TriangleSubsetType::None>, {}};
  } else if (area > 0.5 - 1e-12) {
    return {tag_v<TriangleSubsetType::All>, {}};
  }

  debug_assert(boost::geometry::is_valid(poly));

  return {tag_v<TriangleSubsetType::Some>, poly};
}
} // namespace generate_data
