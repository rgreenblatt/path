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

// can't be InitialTriAndDir
using SampleValue = TaggedUnion<SampleSpecType, eigen_wrapper::Affine3f,
                                Span<const intersect::Ray>, MetaTuple<>>;

template <rng::RngState R>
ATTR_PURE_NDEBUG HOST_DEVICE inline integrate::FRayRayInfo
initial_ray_sample(R &rng, unsigned x, unsigned y, unsigned x_dim,
                   unsigned y_dim, const SampleValue &sample) {
  auto ray =
      sample.visit_tagged([&](auto tag, const auto &spec) -> intersect::Ray {
        if constexpr (tag == SampleSpecType::SquareImage) {
          float x_offset = rng.next();
          float y_offset = rng.next();

          return initial_ray(x + x_offset, y + y_offset, x_dim, y_dim, spec);
        } else if constexpr (tag == SampleSpecType::InitialRays) {
          return spec[x];
        } else {
          static_assert(tag == SampleSpecType::InitialIdxAndDir);
          unreachable();
        }
      });

  return {ray, {.multiplier = 1.f, .target_distance = std::nullopt}};
}

template <std::copyable InfoType>
ATTR_PURE_NDEBUG HOST_DEVICE inline integrate::IntersectionInfo<InfoType>
initial_intersection_sample(const InitialIdxAndDirSpec &initial,
                            Span<const InfoType> idx_to_info) {
  return {
      .intersection =
          {
              // by convention
              .intersection_dist = 1.f,
              // also by convention (change?, important?)
              .is_back_intersection = false,
              .info = idx_to_info[initial.idx],
          },
      .info = {initial.ray,
               {.multiplier = 1.f, .target_distance = std::nullopt}},
  };
}
} // namespace integrate_image
} // namespace detail
} // namespace render
