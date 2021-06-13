#include "generate_data/full_scene/intersect_for_baryocentric_coords.h"

#include "generate_data/full_scene/default_film_to_world.h"
#include "intersect/accel/sbvh/sbvh.h"
#include "intersect/accel/sbvh/sbvh_impl.h"
#include "intersect/triangle_impl.h"
#include "intersectable_scene/flat_triangle/flat_triangle.h"
#include "render/detail/integrate_image/initial_ray_sample.h"
#include "rng/uniform/uniform.h"

namespace generate_data {
namespace full_scene {
IntersectedBaryocentricCoords
intersect_for_baryocentric_coords(const scene::Scene &scene, unsigned dim) {
  IntersectedBaryocentricCoords out;

  intersectable_scene::flat_triangle::Generator<
      ExecutionModel::CPU, intersect::accel::sbvh::Settings,
      intersect::accel::sbvh::SBVH<ExecutionModel::CPU>>
      gen;
  const auto scene_items = gen.gen({}, scene);
  auto film_to_world = default_film_to_world();

  VectorT<unsigned> info_idx_to_orig(scene.triangles().size());

  for (unsigned i = 0; i < scene.triangles().size(); ++i) {
    info_idx_to_orig[scene_items.orig_triangle_idx_to_info[i].idx] = i;
  }

  for (unsigned y = 0; y < dim; ++y) {
    for (unsigned x = 0; x < dim; ++x) {
      const auto ray = render::detail::integrate_image::initial_ray(
          x, y, dim, dim, film_to_world);
      const auto intersection =
          scene_items.intersectable_scene.intersector.intersect(ray);

      if (intersection.has_value()) {
        unsigned idx = info_idx_to_orig[intersection->info.idx];
        const auto point = intersection->intersection_point(ray);
        auto [s, t] = scene.triangles()[idx].baryo_values(point);

        out.coords.push_back({s, t});
        out.tri_idxs.push_back(idx);
        out.directions.push_back(ray.direction);
        out.image_indexes.push_back({x, y});
      }
    }
  }

  return out;
}
} // namespace full_scene
} // namespace generate_data
