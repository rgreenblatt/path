#pragma once

#include "integrate/rendering_equation_state.h"
#include "intersect/ray.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/optional.h"
#include "render/renderer.h"
#include "rng/rng.h"

#include <Eigen/Geometry>

namespace render {
namespace detail {
namespace integrate_image {
ATTR_PURE_NDEBUG HOST_DEVICE inline intersect::Ray
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

using SampleValue = TaggedUnion<SampleSpecType, eigen_wrapper::Affine3f,
                                Span<const intersect::Ray>>;

template <rng::RngState R>
ATTR_PURE_NDEBUG HOST_DEVICE inline integrate::FRayRayInfo
initial_ray_sample(R &rng, unsigned x, unsigned y, unsigned x_dim,
                   unsigned y_dim, const SampleValue &sample) {

  auto ray = sample.visit_tagged([&](auto tag, const auto &spec) {
    if constexpr (tag == SampleSpecType::SquareImage) {

      float x_offset = rng.next();
      float y_offset = rng.next();

      return initial_ray(x + x_offset, y + y_offset, x_dim, y_dim, spec);
    } else {
      static_assert(tag == SampleSpecType::InitialRays);

      return spec[x];
    }
  });

  float multiplier = 1.f;
  return {ray, {multiplier, std::nullopt}};
}
} // namespace integrate_image
} // namespace detail
} // namespace render
