#include "generate_data/shadowed.h"

#include "generate_data/clip_by_plane.h"
#include "generate_data/get_points_from_subset.h"
#include "generate_data/triangle_subset.h"
#include "generate_data/triangle_subset_intersection.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/array_vec.h"
#include "lib/span.h"

#include <boost/geometry.hpp>
#include <unordered_set>

namespace std {
// from
// https://codereview.stackexchange.com/questions/171999/specializing-stdhash-for-stdarray
template <class T, size_t N> struct hash<array<T, N>> {
  auto operator()(const array<T, N> &key) const {
    size_t result = 0;
    for (size_t i = 0; i < N; ++i) {
      result = result * 31 + hasher(key[i]);
    }
    return result;
  }

  std::hash<T> hasher;
};
} // namespace std

namespace generate_data {

// TODO: fix epsilons
// TODO: could be sooooo much faster probably
ATTR_PURE_NDEBUG PartiallyShadowedInfo partially_shadowed(
    const intersect::TriangleGen<double> &from,
    const TriangleSubset &from_clipped_region,
    const intersect::TriangleGen<double> &blocker,
    const TriangleSubset &blocker_clipped_region,
    const intersect::TriangleGen<double> &onto, bool flip_onto_normal) {
  auto from_pb = get_points_from_subset_with_baryo(from, from_clipped_region);
  const auto blocker_pb =
      get_points_from_subset_with_baryo(blocker, blocker_clipped_region);
  const auto &from_vertices = from_pb.points;
  const auto &from_baryo = from_pb.baryo;
  const auto &blocker_vertices = blocker_pb.points;
  const auto &blocker_baryo = blocker_pb.baryo;
  debug_assert(from_vertices.size() == from_baryo.size());
  debug_assert(blocker_vertices.size() == blocker_baryo.size());
  debug_assert(!from_vertices.empty());
  debug_assert(!blocker_vertices.empty());

  auto normal = *onto.normal();
  if (flip_onto_normal) {
    normal = -normal;
  }
  auto plane_vertex = onto.vertices[0];

  double plane_offset = normal.dot(plane_vertex);
  auto plane_pos = [&](const Eigen::Vector3d &point) {
    return point.dot(normal) - plane_offset;
  };

#ifndef NDEBUG
  double from_max_plane_pos = std::numeric_limits<double>::lowest();
  for (const auto &p : from_vertices) {
    double pos = plane_pos(p);
    // should already be clipped by plane!
    debug_assert(pos > -1e-6);
    from_max_plane_pos = std::max(pos, from_max_plane_pos);
  }

  // shouldn't be coplaner
  debug_assert(from_max_plane_pos > 1e-13);

  for (const auto &p : blocker_vertices) {
    // should already be clipped by plane!
    [[maybe_unused]] double pos = plane_pos(p);
    debug_assert(pos > -1e-6);
  }
#endif

  VectorT<Eigen::Vector2d> points_for_hull;
  // TODO: could be handled just using an angle range
  VectorT<Eigen::Vector2d> directions;
  points_for_hull.reserve(from_vertices.size() * blocker_vertices.size() * 6);
  directions.reserve(from_vertices.size() * blocker_vertices.size());

  VectorT<PartiallyShadowedInfo::RayItem> ray_items;
  ray_items.reserve(from_vertices.size() * blocker_vertices.size() * 6);

  std::unordered_set<std::array<double, 4>> added_items;

  auto run_for_points =
      [&](const BaryoPoint &baryo_origin, const BaryoPoint &baryo_endpoint,
          const Eigen::Vector3d &origin, const Eigen::Vector3d &endpoint) {
        std::array key{baryo_origin.x(), baryo_origin.y(), baryo_endpoint.x(),
                       baryo_endpoint.y()};
        if (!added_items.insert(key).second) {
          // already added
          return;
        }

        auto add_point = [&](const Eigen::Vector3d &point) -> Eigen::Vector2d {
          auto baryo = onto.baryo_values(point);
          Eigen::Vector2d baryo_eigen{baryo[0], baryo[1]};
          points_for_hull.push_back(baryo_eigen);

          return baryo_eigen;
        };

        double origin_plane_position = plane_pos(origin);
        double endpoint_plane_position = plane_pos(endpoint);

        debug_assert(origin_plane_position - endpoint_plane_position > -1e-6);

        Eigen::Vector3d direction = endpoint - origin;

        // TODO: handle intersecting/overlapping point case
        debug_assert(direction.norm() > 1e-10);

        auto result = [&]() -> PartiallyShadowedInfo::RayItem::Result {
          if (origin_plane_position - endpoint_plane_position < 1e-6) {
            // coplanar case
            if (std::abs(endpoint_plane_position) < 1e-6) {
              // TODO: is this case important?
              // endpoint is also on onto
              add_point(endpoint);
            }

            auto origin_on_plane_baryo =
                onto.baryo_values(origin - normal * origin_plane_position);
            auto endpoint_on_plane =
                onto.baryo_values(endpoint - normal * endpoint_plane_position);
            Eigen::Vector2d direction{
                endpoint_on_plane[0] - origin_on_plane_baryo[0],
                endpoint_on_plane[1] - origin_on_plane_baryo[1]};
            direction.normalize();
            directions.push_back(direction);
            return {tag_v<RayItemResultType::Ray>, direction};
          } else {
            double denom = normal.dot(direction);
            debug_assert(std::abs(denom) > 1e-15);
            double t = -origin_plane_position / denom;
            debug_assert(t > 1. - 1e-3); // should hit plane AFTER endpoint
            return {tag_v<RayItemResultType::Intersection>,
                    add_point(t * direction + origin)};
          }
        }();

        ray_items.push_back({
            .baryo_origin = baryo_origin,
            .baryo_endpoint = baryo_endpoint,
            .origin = origin,
            .endpoint = endpoint,
            .result = result,
        });
      };

  // TODO: could make this much more efficient in many different ways...
  // Also, this has pretty terrible numerical properties...
  for (unsigned i = 0; i < from_vertices.size(); ++i) {
    auto new_blocker_region = triangle_subset_intersection(
        blocker_clipped_region,
        clip_by_plane(-normal, from_vertices[i].dot(-normal), blocker));

    const auto new_blocker_bp =
        get_points_from_subset_with_baryo(blocker, new_blocker_region);
    const auto &new_blocker_vertices = new_blocker_bp.points;
    const auto &new_blocker_baryo = new_blocker_bp.baryo;
    debug_assert(new_blocker_vertices.size() == new_blocker_baryo.size());

    for (unsigned j = 0; j < new_blocker_vertices.size(); ++j) {
      run_for_points(from_baryo[i], new_blocker_baryo[j], from_vertices[i],
                     new_blocker_vertices[j]);
    }
  }
  for (unsigned j = 0; j < blocker_vertices.size(); ++j) {
    auto new_from_region = triangle_subset_intersection(
        from_clipped_region,
        clip_by_plane(normal, blocker_vertices[j].dot(normal), from));

    const auto new_from_bp =
        get_points_from_subset_with_baryo(from, new_from_region);
    const auto &new_from_vertices = new_from_bp.points;
    const auto &new_from_baryo = new_from_bp.baryo;
    debug_assert(new_from_vertices.size() == new_from_baryo.size());

    for (unsigned i = 0; i < new_from_vertices.size(); ++i) {
      run_for_points(new_from_baryo[i], blocker_baryo[j], new_from_vertices[i],
                     blocker_vertices[j]);
    }
  }

  boost::geometry::model::multi_point<BaryoPoint> multi_point_for_hull;
  auto add_final_point = [&](const Eigen::Vector2d &p) {
    boost::geometry::append(multi_point_for_hull, BaryoPoint{p.x(), p.y()});
  };
  for (const auto &point : points_for_hull) {
    add_final_point(point);
  }

  // this is jank and slow - probably could do this better...
  // Also has numerical issues...
  double edge_total = 0.;
  for (unsigned i = 0; i < onto.vertices.size(); ++i) {
    unsigned next_i = (i + 1) % onto.vertices.size();
    edge_total += (onto.vertices[next_i] - onto.vertices[i]).norm();
  }
  double base_multiplier = edge_total + 2.;
  for (const auto &dir : directions) {
    for (const auto &point : points_for_hull) {
      auto dist_from_0 = point.norm();
      // multiplier just needs to be sufficiently large
      double multiplier = 2. * (dist_from_0 + base_multiplier);
      add_final_point(point + dir * multiplier);
    }
  }

  TriPolygon poly;
  boost::geometry::convex_hull(multi_point_for_hull, poly);
  TriPolygon triangle{{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}}};
  debug_assert(boost::geometry::is_valid(triangle));

  auto partially_shadowed =
      triangle_subset_intersection({tag_v<TriangleSubsetType::Some>, poly},
                                   {tag_v<TriangleSubsetType::Some>, triangle});
  if (partially_shadowed.type() == TriangleSubsetType::Some) {
    auto poly = partially_shadowed.get(tag_v<TriangleSubsetType::Some>);
    if (std::abs(boost::geometry::area(poly) - 0.5) < 1e-12) {
      partially_shadowed = {tag_v<TriangleSubsetType::All>, {}};
    }
  }

  return {.partially_shadowed = partially_shadowed, .ray_items = ray_items};
}

ATTR_PURE_NDEBUG TriangleSubset
shadowed_from_point(const Eigen::Vector3d &point,
                    SpanSized<const Eigen::Vector3d> blocker_points,
                    const intersect::TriangleGen<double> &onto) {
  TriangleSubset full_intersection = {tag_v<TriangleSubsetType::All>, {}};
  debug_assert(blocker_points.size() >= 3);

  for (unsigned j = 0; j < blocker_points.size(); ++j) {
    unsigned next_j = (j + 1) % blocker_points.size();
    unsigned next_next_j = (j + 2) % blocker_points.size();

    auto vec_0 = blocker_points[j] - point;
    auto vec_1 = blocker_points[next_j] - point;
    Eigen::Vector3d normal = vec_0.cross(vec_1);
    auto point_on_plane = blocker_points[next_j];
    // other vertex should be on positive side of plane (blocker_points is
    // assumed to be convex, planar polygon)
    if (normal.dot(blocker_points[next_next_j] - point_on_plane) < 0.f) {
      normal *= -1.f;
    }

    auto clipped = clip_by_plane_point(normal, point_on_plane, onto);

    full_intersection =
        triangle_subset_intersection(full_intersection, clipped);
  }

  return full_intersection;
}

ATTR_PURE_NDEBUG TotallyShadowedInfo
totally_shadowed(const intersect::TriangleGen<double> &from,
                 const TriangleSubset &from_clipped_region,
                 const intersect::TriangleGen<double> &blocker,
                 const TriangleSubset &blocker_clipped_region,
                 const intersect::TriangleGen<double> &onto) {
  // no need for the func to be called in these cases
  always_assert(from_clipped_region.type() != TriangleSubsetType::None);
  always_assert(blocker_clipped_region.type() != TriangleSubsetType::None);

  auto from_points = get_points_from_subset(from, from_clipped_region);
  auto blocker_points = get_points_from_subset(blocker, blocker_clipped_region);
  VectorT<TriangleSubset> from_each_point(from_points.size());
  TriangleSubset totally_shadowed = {tag_v<TriangleSubsetType::All>, {}};
  for (unsigned i = 0; i < from_points.size(); ++i) {
    const auto &origin = from_points[i];
    from_each_point[i] = shadowed_from_point(origin, blocker_points, onto);
    totally_shadowed =
        triangle_subset_intersection(totally_shadowed, from_each_point[i]);
  }

  return {
      .totally_shadowed = totally_shadowed,
      .from_each_point = from_each_point,
  };
}
} // namespace generate_data
