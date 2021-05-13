#pragma once

#include "generate_data/scene_triangles.h"
#include "intersect/is_triangle_between.h"
#include "intersect/is_triangle_blocking.h"
#include "intersect/triangle.h"
#include "scene/material.h"
#include "scene/triangle_constructor.h"

namespace generate_data {
template <rng::RngState R> SceneTriangles generate_scene_triangles(R &rng) {
  auto random_vec = [&]() {
    return Eigen::Vector3f{rng.next(), rng.next(), rng.next()};
  };

  auto gen_tri = [&]() -> intersect::Triangle {
    return {{
        random_vec(),
        random_vec(),
        random_vec(),
    }};
  };

  intersect::Triangle triangle_onto;
  intersect::Triangle triangle_blocking;
  intersect::Triangle triangle_light;

  bool found = false;
  unsigned count = 0;
  while (!found) {
  start:
    ++count;
    std::array<intersect::Triangle, 3> triangles{gen_tri(), gen_tri(),
                                                 gen_tri()};

    for (unsigned i = 0; i < triangles.size(); ++i) {
      for (unsigned other_triangle = (i + 1) % 3; other_triangle != i;
           other_triangle = (other_triangle + 1) % 3) {
        for (unsigned j = 0; j < 3; ++j) {
          unsigned next_j = (j + 1) % 3;
          auto origin = triangles[i].vertices[j];
          Eigen::Vector3f dir = triangles[i].vertices[next_j] - origin;
          float dist = dir.norm();

          auto intersection =
              triangles[other_triangle].intersect(intersect::Ray{
                  .origin = origin,
                  .direction = UnitVector::new_normalize(dir),
              });

          if (intersection.has_value() &&
              intersection->intersection_dist >= 0.f &&
              intersection->intersection_dist <= dist) {
            // triangles intersect!
            goto start;
          }
        }
      }
    }

    for (unsigned inner = 0; inner < triangles.size(); ++inner) {
      unsigned outer_first = (inner + 1) % triangles.size();
      unsigned outer_second = (inner + 2) % triangles.size();

      std::array<intersect::Triangle, 2> outer{triangles[outer_first],
                                               triangles[outer_second]};

      if (!intersect::is_triangle_between(outer, triangles[inner]) ||
          intersect::is_triangle_blocking(outer, triangles[inner])) {
        continue;
      }

      found = true;

      triangle_onto = triangles[outer_first];
      triangle_blocking = triangles[inner];
      triangle_light = triangles[outer_second];

      break;
    }
  }

  return {
      .triangle_onto = triangle_onto,
      .triangle_blocking = triangle_blocking,
      .triangle_light = triangle_light,
  };
}
} // namespace generate_data
