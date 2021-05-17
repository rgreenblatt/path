#include "generate_data/normalize_scene_triangles.h"
#include "generate_data/get_dir_towards.h"
#include "lib/projection.h"

#include "dbg.h"

namespace generate_data {
SceneTriangles normalize_scene_triangles(const SceneTriangles &tris) {
  // this is obviously not the most efficient way of doing this, but
  // it shouldn't matter...
  SceneTrianglesGen<double> new_tris =
      tris.template apply_gen<double>([&](const intersect::Triangle &tri) {
        return tri.template apply_gen<double>([&](const Eigen::Vector3f &vec) {
          return vec.template cast<double>();
        });
      });
  new_tris = new_tris.apply([&](const intersect::TriangleGen<double> &tri) {
    return tri.apply([&](const Eigen::Vector3d &vec) {
      return vec / std::sqrt(new_tris.triangle_onto.area());
    });
  });

  auto dir_away = -get_dir_towards(new_tris);

  auto desired_normal = UnitVectorGen<double>::new_normalize({0., 0., 1.});
  auto desired_first_point = UnitVectorGen<double>::new_normalize({1., 0., 0.});

  auto rotate_to_vert = find_rotate_vector_to_vector(dir_away, desired_normal);

  new_tris = new_tris.apply_transform(rotate_to_vert);
  auto centroid = new_tris.triangle_onto.centroid();
  new_tris = new_tris.apply_transform(Eigen::Translation3d{-centroid});

  auto rotate_to_x = find_rotate_vector_to_vector(
      UnitVectorGen<double>::new_normalize(new_tris.triangle_onto.vertices[0]),
      desired_first_point);

  new_tris = new_tris.apply_transform(rotate_to_x);

  auto apply_sort = [](intersect::TriangleGen<double> &tri) {
    std::sort(tri.vertices.begin(), tri.vertices.end(),
              [&](const Eigen::Vector3d &l, const Eigen::Vector3d &r) {
                return l.z() < r.z();
              });
  };

  apply_sort(new_tris.triangle_light);
  apply_sort(new_tris.triangle_blocking);

  return new_tris.template apply_gen<float>(
      [&](const intersect::TriangleGen<double> &tri) {
        return tri.template apply_gen<float>([&](const Eigen::Vector3d &vec) {
          return vec.template cast<float>();
        });
      });
}
} // namespace generate_data
