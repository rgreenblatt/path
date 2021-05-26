#pragma once

#include "intersect/triangle.h"
#include "lib/attribute.h"

#include <array>

namespace generate_data {
ATTR_PURE_NDEBUG inline bool possibly_shadowed(
    std::array<const intersect::TriangleGen<double> *, 2> endpoints,
    const intersect::TriangleGen<double> &blocker) {
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

  // TODO: maybe more?
  return true;
}
} // namespace generate_data
