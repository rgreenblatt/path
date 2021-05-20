#include "generate_data/shadowed.h"

#include "generate_data/clip_by_plane.h"
#include "generate_data/triangle_subset.h"
#include "generate_data/triangle_subset_convex_union.h"
#include "generate_data/triangle_subset_intersection.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/array_vec.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/multi_point.hpp>

#include "dbg.h"

namespace generate_data {
ATTR_PURE_NDEBUG static ArrayVec<BaryoPoint, 4>
intersecting_points(Eigen::Vector3d origin,
                    const intersect::TriangleGen<double> &blocker,
                    const intersect::TriangleGen<double> &onto) {
  // TODO: consider precomputing some of these values later...
  // We mostly don't need the normal vector to be unit vec...
  auto normal = *onto.normal();
  auto get_numerator = [&]() { return normal.dot(onto.vertices[0] - origin); };
  double numerator = get_numerator();

  // TODO: check these numbers...
  if (std::abs(numerator) < 1e-10) {
    dbg("IN PLANE!");
    // we are in the plane, so we will "push" ourselves outside just a bit
    origin += normal * 1e-6;
    dbg(numerator);
    numerator = get_numerator();
    dbg(numerator);
  }

  std::array<std::optional<BaryoPoint>, 3> op_points;

  auto run_for_blocker_point =
      [&](const Eigen::Vector3d &point) -> std::optional<BaryoPoint> {
    Eigen::Vector3d direction = point - origin;
    double denom = normal.dot(direction);
    dbg(direction);
    dbg(point);
    dbg(origin);
    dbg(normal);
    if (std::abs(denom) < 1e-15) {
      dbg("denom too low");
      return std::nullopt;
    }
    dbg(denom);
    double t = numerator / denom;
    if (t < 0.) {
      dbg("t < 0");
      return std::nullopt;
    }
    auto baryocentric_vals = onto.interpolation_values(t * direction + origin);
    return BaryoPoint{baryocentric_vals[1], baryocentric_vals[2]};
  };

  for (unsigned i = 0; i < 3; ++i) {
    op_points[i] = run_for_blocker_point(blocker.vertices[i]);
  }

  ArrayVec<unsigned, 3> included_idxs;
  ArrayVec<unsigned, 3> excluded_idxs;

  for (unsigned i = 0; i < 3; ++i) {
    if (op_points[i].has_value()) {
      included_idxs.push_back(i);
    } else {
      excluded_idxs.push_back(i);
    }
  }

  ArrayVec<BaryoPoint, 4> out;
  for (unsigned included_idx : included_idxs) {
    out.push_back(*op_points[included_idx]);
  }

  dbg(unsigned(included_idxs.size()));
  if (included_idxs.size() == 3 || included_idxs.empty()) {
    return out;
  }

  auto get_point = [&](unsigned included_idx, unsigned excluded_idx) {
    Eigen::Vector3d toward_included =
        blocker.vertices[included_idx] - blocker.vertices[excluded_idx];
    // the numerical stability of this might be problematic
    auto to_project = (1 + 1e-15) * normal *
                      normal.dot(blocker.vertices[excluded_idx] - origin) /
                      normal.squaredNorm();
    double prop =
        toward_included.dot(to_project) / toward_included.squaredNorm();
    auto out = run_for_blocker_point(blocker.vertices[excluded_idx] +
                                     toward_included * prop);
    dbg(out.has_value());
    dbg(out->x());
    dbg(out->y());

    return *out;
  };

  dbg(included_idxs.size());
  if (included_idxs.size() == 1) {
    out.push_back(get_point(included_idxs[0], excluded_idxs[0]));
    out.push_back(get_point(included_idxs[0], excluded_idxs[1]));
  } else {
    debug_assert(included_idxs.size() == 2);
    out.push_back(get_point(included_idxs[0], excluded_idxs[0]));
    out.push_back(get_point(included_idxs[1], excluded_idxs[0]));
  }

  return out;
}

ATTR_PURE_NDEBUG TriangleSubset shadowed_from_point(
    const Eigen::Vector3d &point, const intersect::TriangleGen<double> &blocker,
    const intersect::TriangleGen<double> &onto) {
  TriangleSubset full_intersection = {tag_v<TriangleSubsetType::All>, {}};

  for (unsigned j = 0; j < blocker.vertices.size(); ++j) {
    unsigned next_j = (j + 1) % blocker.vertices.size();
    unsigned next_next_j = (j + 2) % blocker.vertices.size();

    auto vec_0 = blocker.vertices[j] - point;
    auto vec_1 = blocker.vertices[next_j] - point;
    Eigen::Vector3d normal = vec_0.cross(vec_1);
    auto point_on_plane = blocker.vertices[next_j];
    // other vertex should be on positive side of plane
    if (normal.dot(blocker.vertices[next_next_j] - point_on_plane) < 0.f) {
      normal *= -1.f;
    }

    auto clipped = clip_by_plane(normal, point_on_plane, onto);

    full_intersection =
        triangle_subset_intersection(full_intersection, clipped);
  }

  return full_intersection;
}

ATTR_PURE_NDEBUG ShadowedInfo
shadowed(const VectorT<Eigen::Vector3d> &from_points,
         const intersect::TriangleGen<double> &blocker,
         const intersect::TriangleGen<double> &onto) {

  VectorT<TriangleSubset> from_each_point(from_points.size());
  TriangleSubset totally_blocked = {tag_v<TriangleSubsetType::All>, {}};
  boost::geometry::model::multi_point<BaryoPoint> points;
  for (unsigned i = 0; i < from_points.size(); ++i) {
    const auto &origin = from_points[i];
    from_each_point[i] = shadowed_from_point(origin, blocker, onto);
    totally_blocked =
        triangle_subset_intersection(totally_blocked, from_each_point[i]);
    auto vert_points = intersecting_points(origin, blocker, onto);
    // dbg("before");
    for (const auto &point : vert_points) {
      // std::cout << "point: [" << point.x() << ", " << point.y() << "]\n";
      boost::geometry::append(points, point);
    }
  }

  TriPolygon poly;
  boost::geometry::convex_hull(points, poly);
  TriPolygon triangle{{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}}};
  debug_assert(boost::geometry::is_valid(triangle));

  auto some_blocking =
      triangle_subset_intersection({tag_v<TriangleSubsetType::Some>, poly},
                                   {tag_v<TriangleSubsetType::Some>, triangle});
  if (some_blocking.type() == TriangleSubsetType::Some) {
    auto poly = some_blocking.get(tag_v<TriangleSubsetType::Some>);
    if (std::abs(boost::geometry::area(poly) - 0.5) < 1e-12) {
      // TODO: check this actually happens!
      some_blocking = {tag_v<TriangleSubsetType::All>, {}};
      dbg("IS ALL");
      dbg(poly.outer().size());
      for (const auto &p : poly.outer()) {
        dbg(p.x(), p.y());
      }
    }
  }

  // auto some_blocking = triangle_subset_convex_union(
  //     {from_each_point[0], from_each_point[1], from_each_point[2]});

  return {
      .some_blocking = some_blocking,
      .totally_blocked = totally_blocked,
      .from_each_point = from_each_point,
  };
}

} // namespace generate_data
