#include "generate_data/normalize_scene_triangles.h"

#include "generate_data/triangle.h"
#include "lib/projection.h"

namespace generate_data {
SceneTriangles normalize_scene_triangles(const SceneTriangles &tris) {
  // this is obviously not the most efficient way of doing this, but
  // it shouldn't matter too much for now...
  auto light_normal = tris.triangle_light.normal_raw();
  auto onto_normal = tris.triangle_onto.normal_raw();

  auto new_tris = tris.apply([&](const intersect::TriangleGen<double> &tri) {
    return tri.apply([&](const Eigen::Vector3d &vec) {
      return vec / std::sqrt(tris.triangle_onto.area());
    });
  });

  auto apply_transform = [&](const auto &transform) {
    new_tris = new_tris.apply_transform(transform);
    light_normal = transform * light_normal;
    onto_normal = transform * onto_normal;
  };

  auto desired_normal = UnitVectorGen<double>::new_normalize({0., 0., 1.});
  auto desired_first_point = UnitVectorGen<double>::new_normalize({1., 0., 0.});

  auto rotate_to_vert = find_rotate_vector_to_vector(
      new_tris.triangle_onto.normal(), desired_normal);

  apply_transform(rotate_to_vert);

  auto centroid = new_tris.triangle_onto.centroid();
  // just apply to tris (not vecs)
  new_tris = new_tris.apply_transform(Eigen::Translation3d{-centroid});

  auto rotate_to_x = find_rotate_vector_to_vector(
      UnitVectorGen<double>::new_normalize(new_tris.triangle_onto.vertices[0]),
      desired_first_point);

  apply_transform(rotate_to_x);

  auto apply_sort = [](intersect::TriangleGen<double> &tri) {
    std::sort(tri.vertices.begin(), tri.vertices.end(),
              [&](const Eigen::Vector3d &l, const Eigen::Vector3d &r) {
                return l.z() < r.z();
              });
  };

  apply_sort(new_tris.triangle_onto);
  apply_sort(new_tris.triangle_blocking);
  apply_sort(new_tris.triangle_light);

  auto restore_normal = [&](Eigen::Vector3d &orig_normal, Triangle &tri) {
    orig_normal.normalize();
    auto actual_normal = *tri.normal();
    auto dotted = orig_normal.dot(actual_normal);
    // should be aligned (dotted is about 1 or -1)
    debug_assert(std::abs(std::abs(dotted) - 1.f) < 1e-10);
    if (std::abs(dotted + 1) < 0.5) {
      // case where vector points away
      std::swap(tri.vertices[1], tri.vertices[2]);
    }
    [[maybe_unused]] auto new_actual_normal = *tri.normal();
    [[maybe_unused]] auto new_dotted = orig_normal.dot(new_actual_normal);
    debug_assert(std::abs(new_dotted - 1.f) < 1e-10);
  };
  restore_normal(onto_normal, new_tris.triangle_onto);
  restore_normal(light_normal, new_tris.triangle_light);

  debug_assert((*new_tris.triangle_onto.normal() - *desired_normal).norm() <
               1e-7);

  return new_tris;
}
} // namespace generate_data
