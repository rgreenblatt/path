#pragma once

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
  ray.direction = (world_space_film_plane - ray.origin).normalized();

  return ray;
}

template <intersectable_scene::IntersectableScene S,
          LightSamplerRef<typename S::B> L, DirSamplerRef<typename S::B> D,
          TermProbRef T, rng::RngRef R>
HOST_DEVICE inline Eigen::Array3f
integrate_pixel(unsigned x, unsigned y, unsigned start_sample,
                unsigned end_sample, const GeneralSettings &settings,
                unsigned x_dim, unsigned y_dim, const S &scene,
                const L &light_sampler, const D &dir_sampler,
                const T &term_prob, const R &rng_ref,
                const Eigen::Affine3f &film_to_world) {
  unsigned sample_idx = start_sample;
  bool finished = true;
  bool count_emission = true;

  intersect::Ray ray;
  Eigen::Array3f multiplier;
  unsigned iters;

  typename R::State rng;

  Eigen::Array3f intensity = Eigen::Array3f::Zero();

  while (!finished || sample_idx != end_sample) {
    if (finished) {
      multiplier = Eigen::Vector3f::Ones();
      rng = rng_ref.get_generator(sample_idx, x, y);
      iters = 0;

      float x_offset = rng.next();
      float y_offset = rng.next();

      ray =
          initial_ray(x + x_offset, y + y_offset, x_dim, y_dim, film_to_world);

      finished = false;
      count_emission = true;

      sample_idx++;
    }

    auto next_intersection_op = scene.intersect(ray);

    if (!next_intersection_op.has_value()) {
      finished = true;
      continue;
    }

    const auto &next_intersection = *next_intersection_op;

    auto include_lighting = [&](const auto &intersection) {
      return !settings.back_cull_emission || !intersection.is_back_intersection;
    };

    const auto &material = scene.get_material(next_intersection);

    if ((!L::performs_samples || count_emission) &&
        include_lighting(next_intersection)) {
      intensity += multiplier * material.emission;
    }

    Eigen::Vector3f normal = scene.get_normal(next_intersection, ray);

    auto intersection_point = next_intersection.intersection_point(ray);

    using B = typename S::B;

    if constexpr (B::continuous) {
      auto add_direct_lighting = [&](float prob_continuous) {
        Eigen::Array3f total = Eigen::Array3f::Zero();
        const auto samples = light_sampler(intersection_point, material,
                                           ray.direction, normal, rng);
        for (unsigned i = 0; i < samples.num_samples; i++) {
          const auto &[dir_sample, expected_distance] = samples.samples[i];

          // TODO: BSDF case
          if (dir_sample.direction.dot(normal) <= 0.f) {
            continue;
          }

          intersect::Ray light_ray{intersection_point, dir_sample.direction};

          auto light_intersection_op = scene.intersect(light_ray);
          if (!light_intersection_op.has_value()) {
            continue;
          }

          const auto &light_intersection = *light_intersection_op;

          // TODO: verify this behaves as expected...
          if (std::abs(light_intersection.intersection_dist -
                       expected_distance) > 1e-6) {
            continue;
          }

          if (!include_lighting(light_intersection)) {
            continue;
          }

          const auto light_multiplier = material.bsdf.continuous_eval(
              ray.direction, light_ray.direction, normal);

          const auto &light_material = scene.get_material(light_intersection);

          total += light_material.emission * light_multiplier *
                   prob_continuous / dir_sample.multiplier;
        }

        intensity += multiplier * total;
      };

      if constexpr (B::discrete) {
        float prob_continuous =
            material.bsdf.prob_continuous(ray.direction, normal);
        if (prob_continuous > 0.f) {
          add_direct_lighting(prob_continuous);
        }
      } else {
        add_direct_lighting(1.f);
      }
    }

    auto sample = dir_sampler(intersection_point, material.bsdf, ray.direction,
                              normal, rng);
    count_emission = sample.is_discrete;

    multiplier *= sample.sample.multiplier;

    auto this_term_prob = term_prob(iters, multiplier);

    if (rng.next() <= this_term_prob) {
      finished = true;
      continue;
    }

    multiplier /= (1.0f - this_term_prob);

    assert(multiplier.x() >= 0.0f);
    assert(multiplier.y() >= 0.0f);
    assert(multiplier.z() >= 0.0f);

    ray.origin = intersection_point;
    ray.direction = sample.sample.direction;
    iters++;
  }

  return intensity;
}
} // namespace detail
} // namespace render
