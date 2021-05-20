#include "generate_data/normalize_scene_triangles.h"

#include "generate_data/get_dir_towards.h"
#include "lib/projection.h"

#include "dbg.h"

namespace generate_data {
SceneTriangles normalize_scene_triangles(const SceneTriangles &tris) {
  // this is obviously not the most efficient way of doing this, but
  // it shouldn't matter too much for now...
  auto new_tris = tris.template cast<double>();
  auto light_normal = new_tris.triangle_light.normal_scaled_by_area();

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
  light_normal = rotate_to_vert * light_normal;
  new_tris = new_tris.apply_transform(Eigen::Translation3d{-centroid});

  auto rotate_to_x = find_rotate_vector_to_vector(
      UnitVectorGen<double>::new_normalize(new_tris.triangle_onto.vertices[0]),
      desired_first_point);

  new_tris = new_tris.apply_transform(rotate_to_x);
  light_normal = rotate_to_x * light_normal;

  auto apply_sort = [](intersect::TriangleGen<double> &tri) {
    std::sort(tri.vertices.begin(), tri.vertices.end(),
              [&](const Eigen::Vector3d &l, const Eigen::Vector3d &r) {
                return l.z() < r.z();
              });
  };

  apply_sort(new_tris.triangle_light);
  apply_sort(new_tris.triangle_blocking);

  light_normal.normalize();
  auto actual_light_normal = *new_tris.triangle_light.normal();
  auto dotted = light_normal.dot(actual_light_normal);
  // should be aligned (dotted is about 1 or -1)
  debug_assert(std::abs(std::abs(dotted) - 1.f) < 1e-10);
  if (std::abs(dotted + 1) < 1e-10) {
    // case where vector points away
    std::swap(new_tris.triangle_light.vertices[1],
              new_tris.triangle_light.vertices[2]);
  }
  [[maybe_unused]] auto new_actual_light_normal =
      *new_tris.triangle_light.normal();
  [[maybe_unused]] auto new_dotted = light_normal.dot(new_actual_light_normal);
  debug_assert(std::abs(new_dotted - 1.f) < 1e-10);

  return new_tris.template cast<float>();
}
} // namespace generate_data
