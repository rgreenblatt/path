#pragma once

#include "integrate/rendering_equation.h"
#include "render/detail/integrate_image.h"

namespace render {
namespace detail {
HOST_DEVICE inline intersect::Ray
initial_ray(float x, float y, unsigned x_dim, unsigned y_dim,
            const Eigen::Affine3f &film_to_world) {
  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * x) / x_dim - 1.0f, (-2.0f * y) / y_dim + 1.0f, -1.0f);
  const auto world_space_film_plane = film_to_world * camera_space_film_plane;

  intersect::Ray ray;

  ray.origin = film_to_world.translation();
  ray.direction =
      UnitVector::new_normalize(world_space_film_plane - ray.origin);

  return ray;
}

template <intersectable_scene::IntersectableScene S,
          LightSamplerRef<typename S::B> L, DirSamplerRef<typename S::B> D,
          TermProbRef T, rng::RngRef R>
HOST_DEVICE inline Eigen::Array3f integrate_pixel(
    unsigned x, unsigned y, unsigned start_sample, unsigned end_sample,
    const integrate::RenderingEquationSettings &settings, unsigned x_dim,
    unsigned y_dim, const S &scene, const L &light_sampler,
    const D &dir_sampler, const T &term_prob, const R &rng_ref,
    const Eigen::Affine3f &film_to_world) {
  auto initial_ray_sampler = [&](auto &rng) -> integrate::RaySampleDistance {
    float x_offset = rng.next();
    float y_offset = rng.next();

    float multiplier = 1.f;
    return {
        initial_ray(x + x_offset, y + y_offset, x_dim, y_dim, film_to_world),
        multiplier, nullopt_value};
  };

  return integrate::rendering_equation(
      initial_ray_sampler, start_sample, end_sample, x + y * x_dim, settings,
      scene, light_sampler, dir_sampler, term_prob, rng_ref);
}
} // namespace detail
} // namespace render
