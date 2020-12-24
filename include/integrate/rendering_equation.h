#pragma once

#include "integrate/dir_sample.h"
#include "integrate/dir_sampler/dir_sampler.h"
#include "integrate/light_sampler/light_sampler.h"
#include "integrate/rendering_equation_settings.h"
#include "integrate/term_prob/term_prob.h"
#include "intersectable_scene/intersectable_scene.h"

namespace integrate {
struct RaySampleDistance {
  intersect::Ray ray;
  float multiplier;
  Optional<float> target_distance;
};

template <typename F, intersectable_scene::IntersectableScene S,
          light_sampler::LightSamplerRef<typename S::B> L,
          dir_sampler::DirSamplerRef<typename S::B> D, term_prob::TermProbRef T,
          rng::RngRef R>
HOST_DEVICE inline Eigen::Array3f
rendering_equation(F &&initial_ray_sampler, unsigned start_sample,
                   unsigned end_sample, unsigned location,
                   const RenderingEquationSettings &settings, const S &scene,
                   const L &light_sampler, const D &dir_sampler,
                   const T &term_prob, const R &rng_ref) {
  unsigned sample_idx = start_sample;
  bool finished = true;
  bool count_emission = true;

  intersect::Ray ray;
  Eigen::Array3f multiplier;
  unsigned iters;

  typename R::State rng;

  Eigen::Array3f intensity = Eigen::Array3f::Zero();
  Optional<float> target_distance;

  while (!finished || sample_idx != end_sample) {
    if (finished) {
      rng = rng_ref.get_generator(sample_idx, location);
      iters = 0;
      RaySampleDistance sample = initial_ray_sampler(rng);
      ray = sample.ray;
      multiplier = Eigen::Vector3f::Constant(sample.multiplier);
      target_distance = sample.target_distance;
      finished = false;
      count_emission = true;
      sample_idx++;
    }

    auto next_intersection_op = scene.intersect(ray);

    auto use_intersection = [&](const auto &intersection,
                                Optional<float> target_distance) {
      return intersection.has_value() &&
             (!target_distance.has_value() ||
              std::abs(*target_distance - intersection->intersection_dist) <
                  1e-6);
    };

    if (!use_intersection(next_intersection_op, target_distance)) {
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
      // TODO generalize
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
          const auto &[dir_sample, light_target_distance] = samples.samples[i];

          // TODO: BSDF case
          if (dir_sample.direction.dot(normal) <= 0.f) {
            continue;
          }

          intersect::Ray light_ray{intersection_point, dir_sample.direction};

          auto light_intersection_op = scene.intersect(light_ray);

          if (!use_intersection(light_intersection_op, light_target_distance) ||
              !include_lighting(*light_intersection_op)) {
            continue;
          }

          const auto light_multiplier = material.bsdf.continuous_eval(
              ray.direction, light_ray.direction, normal);

          const auto &light_material =
              scene.get_material(*light_intersection_op);

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
} // namespace integrate
