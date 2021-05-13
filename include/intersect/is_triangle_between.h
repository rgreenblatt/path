#pragma once

#include "intersect/triangle.h"

namespace intersect {
HOST_DEVICE inline bool is_triangle_between(const std::array<Triangle, 2> &tris,
                                            const Triangle &possibly_between) {
  // TODO: there is probably a more efficient way of doing this...
  // And/or use filtering where possible...
  // TODO: check totally blocked case?
  float intersection_surface_area =
      tris[0]
          .bounds()
          .union_other(tris[1].bounds())
          .intersection_other(possibly_between.bounds())
          .surface_area();
  if (intersection_surface_area < 0.f) {
    return false;
  }

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      auto origin = tris[0].vertices[i];
      auto dir = tris[1].vertices[j] - origin;
      float distance = dir.norm();

      auto intersection = possibly_between.intersect(intersect::Ray{
          .origin = origin,
          .direction = UnitVector::new_normalize(dir),
      });
      if (intersection.has_value() && intersection->intersection_dist >= 0.f &&
          intersection->intersection_dist <= distance) {
        return true;
      }

      for (unsigned tri_idx = 0; tri_idx < tris.size(); ++tri_idx) {
        unsigned other_idx = (tri_idx + 1) % tris.size();

        auto origin = tris[tri_idx].vertices[i];
        Eigen::Vector3f dir = possibly_between.vertices[j] - origin;
        float distance = dir.norm();

        auto intersection = tris[other_idx].intersect(intersect::Ray{
            .origin = origin,
            .direction = UnitVector::new_normalize(dir),
        });
        if (intersection.has_value() &&
            intersection->intersection_dist >= 0.f &&
            intersection->intersection_dist >= distance) {
          return true;
        }

        {
          unsigned next_i = (i + 1) % 3;
          intersect::Triangle tri{{
              tris[tri_idx].vertices[i],
              tris[tri_idx].vertices[next_i],
              tris[other_idx].vertices[j],
          }};
          for (unsigned k = 0; k < 3; ++k) {
            unsigned next_k = (k + 1) % 3;
            auto origin = possibly_between.vertices[k];
            auto dir = possibly_between.vertices[next_k] - origin;
            float distance = dir.norm();

            auto intersection = tri.intersect(intersect::Ray{
                .origin = origin,
                .direction = UnitVector::new_normalize(dir),
            });
            if (intersection.has_value() &&
                intersection->intersection_dist >= 0.f &&
                intersection->intersection_dist <= distance) {
              return true;
            }
          }
        }
      }
    }
  }
  for (unsigned i = 0; i < 3; ++i) {
    unsigned next = (i + 1) % 3;
    auto origin = possibly_between.vertices[i];
    auto dir = possibly_between.vertices[next] - origin;
    float distance = dir.norm();
    intersect::Ray ray{
        .origin = origin,
        .direction = UnitVector::new_normalize(dir),
    };

    for (const auto &tri : tris) {
      auto intersection = tri.intersect(ray);
      if (intersection.has_value() && intersection->intersection_dist >= 0.f &&
          intersection->intersection_dist <= distance) {
        return true;
      }
    }
  }

  return false;
}
} // namespace intersect
