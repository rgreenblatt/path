#include "generate_data/single_triangle/generate_scene_triangles.h"

#include "generate_data/possibly_shadowed.h"
#include "generate_data/triangle.h"
#include "intersect/triangle_impl.h"

namespace generate_data {
namespace single_triangle {
SceneTriangles generate_scene_triangles(UniformState &rng) {
  auto random_vec = [&]() {
    return Eigen::Vector3d{rng.next(), rng.next(), rng.next()};
  };

  auto gen_tri = [&](float z_offset) -> Triangle {
    Eigen::Vector3d addr = Eigen::Vector3d::UnitZ() * z_offset;
    return {{
        random_vec() + addr,
        random_vec() + addr,
        random_vec() + addr,
    }};
  };

  while (true) {
  start:
    std::array<Triangle, 3> triangles{gen_tri(0.), gen_tri(0.5), gen_tri(1.)};

    // avoid intersection (may be relaxed later...)
    for (unsigned i = 0; i < triangles.size(); ++i) {
      for (unsigned other_triangle = (i + 1) % 3; other_triangle != i;
           other_triangle = (other_triangle + 1) % 3) {
        for (unsigned j = 0; j < 3; ++j) {
          unsigned next_j = (j + 1) % 3;
          auto origin = triangles[i].vertices[j];
          Eigen::Vector3d dir = triangles[i].vertices[next_j] - origin;
          float dist = dir.norm();

          auto intersection =
              triangles[other_triangle].intersect(intersect::GenRay<double>{
                  .origin = origin,
                  .direction = UnitVectorGen<double>::new_normalize(dir),
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

      // make normals point toward each other
      Eigen::Vector3d centroid_vec = triangles[outer_second].centroid() -
                                     triangles[outer_first].centroid();
      if (centroid_vec.dot(triangles[outer_first].normal_raw()) < 0.) {
        // flip normal
        std::swap(triangles[outer_first].vertices[1],
                  triangles[outer_first].vertices[2]);
      }
      if (-centroid_vec.dot(triangles[outer_second].normal_raw()) < 0.) {
        // flip normal
        std::swap(triangles[outer_second].vertices[1],
                  triangles[outer_second].vertices[2]);
      }

      if (!possibly_shadowed(
              {&triangles[outer_first], &triangles[outer_second]},
              triangles[inner])) {
        continue;
      }

      return {
          .triangle_onto = triangles[outer_first],
          .triangle_blocking = triangles[inner],
          .triangle_light = triangles[outer_second],
      };
    }
  }
}
} // namespace single_triangle
} // namespace generate_data
