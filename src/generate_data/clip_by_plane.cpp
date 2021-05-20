#include "generate_data/clip_by_plane.h"

#include "generate_data/triangle_subset.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/array_vec.h"

#include <boost/geometry.hpp>

namespace generate_data {
ATTR_PURE_NDEBUG TriangleSubset
clip_by_plane(const Eigen::Vector3d &normal, const Eigen::Vector3d &point,
              const intersect::TriangleGen<double> &tri) {
  ArrayVec<unsigned, 3> included;
  ArrayVec<double, 3> precomputed_vals;
  ArrayVec<unsigned, 3> excluded;
  double min_v = normal.dot(point);
  for (unsigned i = 0; i < 3; ++i) {
    double dotted = normal.dot(tri.vertices[i]);
    if (dotted > min_v) {
      precomputed_vals.push_back(min_v - dotted);
      included.push_back(i);
    } else {
      excluded.push_back(i);
    }
  }

  always_assert(included.size() + excluded.size() == 3);

  if (included.size() == 3) {
    return {tag_v<TriangleSubsetType::All>, {}};
  }
  if (excluded.size() == 3) {
    return {tag_v<TriangleSubsetType::None>, {}};
  }

  auto make_point = [&](double value, unsigned idx) -> Eigen::Vector2d {
    if (idx == 2) {
      return {0., value};
    } else {
      debug_assert(idx == 1);
      return {value, 0.};
    }
  };

  // auto

  auto edge_points = [&](unsigned included_idx, unsigned end_idx) {
    unsigned origin_idx = included[included_idx];
    Eigen::Vector3d origin = tri.vertices[origin_idx];
    Eigen::Vector3d ray = tri.vertices[end_idx] - origin;

    // NOTE: ray.dot(normal) can't be zero (outside of some serious floating
    // point issues) because that would imply that either all points are
    // included or all are excluded.
    double prop = precomputed_vals[included_idx] / ray.dot(normal);
    Eigen::Vector3d endpoint = prop * ray + origin;

    debug_assert(std::abs(endpoint.dot(normal) - min_v) < 1e-6);

    if (origin_idx == 0) {
      return make_point(prop, end_idx);
    } else if (end_idx == 0) {
      return make_point(1 - prop, origin_idx);
    } else {
      auto baryocentric_vals = tri.interpolation_values(endpoint);
      Eigen::Vector2d ret{baryocentric_vals[1], baryocentric_vals[2]};
      [[maybe_unused]] Eigen::Vector3d baryo_point =
          tri.vertices[0] + ret.x() * (tri.vertices[1] - tri.vertices[0]) +
          ret.y() * (tri.vertices[2] - tri.vertices[0]);
      debug_assert((endpoint - baryo_point).norm() < 1e-6);

      return ret;
    }
  };

  ArrayVec<Eigen::Vector2d, 4> points;

  auto add_point = [&](const Eigen::Vector2d &p) { points.push_back(p); };
  if (included.size() == 2) {
    add_point(edge_points(0, excluded[0]));
    add_point(edge_points(1, excluded[0]));
  } else {
    always_assert(included.size() == 1);
    add_point(edge_points(0, excluded[0]));
    add_point(edge_points(0, excluded[1]));
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

  if (boost::geometry::area(poly) <= 1e-15) {
    return {tag_v<TriangleSubsetType::None>, {}};
  }

  debug_assert(boost::geometry::is_valid(poly));
  debug_assert(boost::geometry::area(poly) <= 0.5);

  return {tag_v<TriangleSubsetType::Some>, poly};
}
} // namespace generate_data
