#pragma once

#include "generate_data/clip_by_plane.h"
#include "generate_data/get_points_from_subset.h"
#include "generate_data/triangle_subset_intersection.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/attribute.h"

#include <array>

namespace generate_data {
// NOTE: normals must be pointed at each other.
ATTR_PURE_NDEBUG inline bool possibly_shadowed(
    std::array<const intersect::TriangleGen<double> *, 2> endpoints,
    const intersect::TriangleGen<double> &blocker,
    std::array<bool, 2> flip_normal = {false, false}) {
  // has to be done in float for now...
  std::array bounds{endpoints[0]->template cast<float>().bounds(),
                    endpoints[1]->template cast<float>().bounds()};
  float intersection_surface_area =
      bounds[0]
          .union_other(bounds[1])
          .intersection_other(blocker.template cast<float>().bounds())
          .surface_area();
  if (intersection_surface_area <= 0.f) {
    return false;
  }

  std::array normals{endpoints[0]->normal(), endpoints[1]->normal()};
  for (unsigned i = 0; i < 2; ++i) {
    if (flip_normal[i]) {
      normals[i] = -normals[i];
    }
  }
  std::array plane_threshold{normals[0]->dot(endpoints[0]->vertices[0]),
                             normals[1]->dot(endpoints[1]->vertices[0])};

  for (unsigned i = 0; i < endpoints.size(); ++i) {
    unsigned other = (i + 1) % endpoints.size();

    double other_max_plane_pos = std::numeric_limits<double>::lowest();
    for (const auto &vert : endpoints[other]->vertices) {
      double plane_pos = vert.dot(*normals[i]) - plane_threshold[i];
      other_max_plane_pos = std::max(other_max_plane_pos, plane_pos);
    }
    if (other_max_plane_pos < 0.) {
      return false; // other is behind
    }
    double blocker_min_plane_pos = std::numeric_limits<double>::max();
    bool point_ahead = false;
    for (const auto &vert : blocker.vertices) {
      double plane_pos = vert.dot(*normals[i]) - plane_threshold[i];
      if (plane_pos > 0.) {
        point_ahead = true;
      }
      blocker_min_plane_pos = std::min(blocker_min_plane_pos, plane_pos);
    }
    if (!point_ahead) {
      return false; // blocker is behind
    }
    if (other_max_plane_pos - blocker_min_plane_pos < 1e-15) {
      return false; // blocker is behind other
    }
  }

  Eigen::Vector3d centroid_vec =
      (endpoints[1]->centroid() - endpoints[0]->centroid()).normalized();

  Eigen::Vector3d axis_0;
  if ((centroid_vec - Eigen::Vector3d::UnitX()).squaredNorm() > 1e-3) {
    axis_0 = centroid_vec.cross(Eigen::Vector3d::UnitX());
  } else {
    axis_0 = centroid_vec.cross(Eigen::Vector3d::UnitY());
  }
  axis_0.normalize();
  Eigen::Vector3d axis_1 = centroid_vec.cross(axis_0).normalized();

  debug_assert(std::abs(axis_0.norm() - 1.) < 1e-8);
  debug_assert(std::abs(axis_1.norm() - 1.) < 1e-8);

  double min_plane_pos = std::numeric_limits<double>::max();
  double max_plane_pos = std::numeric_limits<double>::lowest();
  Eigen::Vector2d min_projected{std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max()};
  Eigen::Vector2d max_projected{std::numeric_limits<double>::lowest(),
                                std::numeric_limits<double>::lowest()};

  auto pos_and_proj = [&](const Eigen::Vector3d &point) {
    double plane_pos = point.dot(centroid_vec);
    Eigen::Vector3d point_on_plane = point - centroid_vec * plane_pos;
    debug_assert(std::abs(point_on_plane.dot(centroid_vec)) < 1e-8);
    Eigen::Vector2d proj{axis_0.dot(point_on_plane),
                         axis_1.dot(point_on_plane)};

    return std::tuple{plane_pos, proj};
  };

  for (unsigned i = 0; i < endpoints.size(); ++i) {
    for (const auto &vert : endpoints[i]->vertices) {
      auto pos_proj = pos_and_proj(vert);
      double plane_pos = std::get<0>(pos_proj);
      auto proj = std::get<1>(pos_proj);
      min_plane_pos = std::min(min_plane_pos, plane_pos);
      max_plane_pos = std::max(max_plane_pos, plane_pos);
      min_projected = min_projected.cwiseMin(proj);
      max_projected = max_projected.cwiseMax(proj);
    }
  }

#if 0
  return true;
#else
  // not sure how efficient this is...
  auto points = get_points_from_subset(
      blocker, triangle_subset_intersection(
                   clip_by_plane(centroid_vec, min_plane_pos, blocker),
                   clip_by_plane(-centroid_vec, -max_plane_pos, blocker)));
  if (points.empty()) {
    return false;
  }

  VectorT<std::tuple<Eigen::Vector2d, double>> points_angles(points.size());
  std::transform(points.begin(), points.end(), points_angles.begin(),
                 [&](const Eigen::Vector3d &point) {
                   return std::tuple{std::get<1>(pos_and_proj(point)), 0.};
                 });
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &[p, ang] : points_angles) {
    centroid += p;
  }
  centroid /= points_angles.size();

  for (auto &[p, ang] : points_angles) {
    Eigen::Vector2d vec = p - centroid;
    ang = std::atan2(vec.y(), vec.x());
  }

  // sort by angle to ensure valid poly
  std::sort(points_angles.begin(), points_angles.end(),
            [](const auto &l, const auto &r) {
              return std::get<1>(l) > std::get<1>(r);
            });

  // not actually on triangle, just using typedef
  TriPolygon poly_on_plane;

  for (const auto &[point, ang] : points_angles) {
    boost::geometry::append(poly_on_plane, eigen_to_baryo(point));
  }
  boost::geometry::append(poly_on_plane,
                          eigen_to_baryo(std::get<0>(points_angles[0])));
  debug_assert(boost::geometry::is_valid(poly_on_plane));

  boost::geometry::model::box<BaryoPoint> box{eigen_to_baryo(min_projected),
                                              eigen_to_baryo(max_projected)};

  return boost::geometry::intersects(poly_on_plane, box);
#endif
}
} // namespace generate_data
