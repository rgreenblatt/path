#pragma once

#include "integrate/dir_sample.h"
#include "integrate/rendering_equation_components.h"
#include "integrate/rendering_equation_settings.h"
#include "integrate/rendering_equation_state.h"
#include "lib/assert.h"
#include "lib/tagged_union.h"
#include "work_division/location_info.h"

namespace integrate {
template <rng::RngState R, ExactSpecializationOf<RenderingEquationComponents> C>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline IterationOutput<C::L::max_sample_size>
rendering_equation_iteration(
    const RenderingEquationState<C::L::max_sample_size> &state, R &rng,
    const ArrayVec<intersect::IntersectionOp<typename C::InfoType>,
                   C::L::max_sample_size + 1> &intersections,
    const RenderingEquationSettings &settings, const C &inp) {
  const auto &[iters, count_emission, has_next_sample, ray_ray_info,
               light_samples, old_intensity] = state;
  const auto &[scene, light_sampler, dir_sampler, term_prob] = inp;

  auto use_intersection = [&](const auto &intersection,
                              Optional<float> target_distance) {
    return intersection.has_value() &&
           (!target_distance.has_value() ||
            std::abs(*target_distance - intersection->intersection_dist) <
                1e-6);
  };

  auto include_lighting = [&](const auto &intersection) {
    // TODO generalize
    return !settings.back_cull_emission || !intersection.is_back_intersection;
  };

  RenderingEquationState<C::L::max_sample_size> new_state;
  new_state.iters = iters + 1;
  new_state.intensity = old_intensity;
  auto &intensity = new_state.intensity;

  debug_assert_assume(light_samples.size() ==
                      intersections.size() - has_next_sample);
  debug_assert_assume(light_samples.size() <= C::L::max_sample_size);
  for (unsigned i = 0; i < light_samples.size(); ++i) {
    const auto &[multiplier, target_distance] = light_samples[i];
    const auto &intersection_op = intersections[i];

    if (!use_intersection(intersection_op, target_distance) ||
        !include_lighting(*intersection_op)) {
      continue;
    }

    intensity += scene.get_material(*intersection_op).emission * multiplier;
  }

  if (!has_next_sample) {
    return {TAG(IterationOutputType::Finished), intensity};
  }

  debug_assert(intersections.size() > 0);

  const auto &ray = ray_ray_info.ray;
  const auto &next_intersection_op = intersections[intersections.size() - 1];

  if (!use_intersection(next_intersection_op,
                        ray_ray_info.info.target_distance)) {
    return {TAG(IterationOutputType::Finished), intensity};
  }

  const auto &next_intersection = *next_intersection_op;
  const auto &intersection_point = next_intersection.intersection_point(ray);
  Eigen::Array3f multiplier = ray_ray_info.info.multiplier;

  decltype(auto) material = scene.get_material(next_intersection);

  if ((!C::L::performs_samples || count_emission) &&
      include_lighting(next_intersection)) {
    intensity += multiplier * material.emission;
  }

  decltype(auto) normal = scene.get_normal(next_intersection, ray);

  using B = typename C::B;

  ArrayVec<intersect::Ray, C::L::max_sample_size + 1> new_rays;

  if constexpr (B::continuous) {
    auto add_direct_lighting = [&, &multiplier =
                                       multiplier](float prob_continuous) {
      const auto samples = inp.light_sampler(intersection_point, material,
                                             ray.direction, normal, rng);
      for (const auto &[dir_sample, light_target_distance] : samples) {
        debug_assert(material.bsdf.is_brdf());
        // TODO: BSDF case
        if (dir_sample.direction->dot(*normal) <= 0.f) {
          continue;
        }

        intersect::Ray light_ray{intersection_point, dir_sample.direction};

        // FIXME: CHECK ME... multiplier
        const auto light_multiplier =
            (multiplier *
             material.bsdf.continuous_eval(ray.direction, light_ray.direction,
                                           normal) *
             prob_continuous / dir_sample.multiplier)
                .eval();

        new_state.light_samples.push_back(
            {light_multiplier, light_target_distance});

        new_rays.push_back(light_ray);
      }
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
  new_state.count_emission = sample.is_discrete;

  Eigen::Array3f new_multiplier = multiplier * sample.sample.multiplier;

  auto this_term_prob = term_prob(iters, new_multiplier);

  new_state.has_next_sample = rng.next() > this_term_prob;

  if (new_state.has_next_sample) {
    new_multiplier /= (1.0f - this_term_prob);

    debug_assert(new_multiplier.x() >= 0.0f);
    debug_assert(new_multiplier.y() >= 0.0f);
    debug_assert(new_multiplier.z() >= 0.0f);

    intersect::Ray new_ray{intersection_point, sample.sample.direction};
    new_state.ray_ray_info = {new_ray, {new_multiplier, nullopt_value}};
    new_rays.push_back(new_ray);
  } else {
    if (new_rays.size() == 0) {
      return {TAG(IterationOutputType::Finished), intensity};
    }
  }

  return {TAG(IterationOutputType::NextIteration), new_state, new_rays};
}

template <typename T>
concept InitialRaySampler = requires(const T &v, rng::MockRngState &rng) {
  { v(rng) } -> std::same_as<FRayRayInfo>;
};

template <InitialRaySampler F, rng::RngRef R, intersect::Intersectable I,
          ExactSpecializationOf<RenderingEquationComponents> C>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline Eigen::Array3f
rendering_equation(const work_division::LocationInfo &location_info,
                   const RenderingEquationSettings &settings,
                   const F &initial_ray_sampler, const R &rng_ref,
                   const I &intersectable, const C &inp) {
  const auto &[start_sample, end_sample, location] = location_info;

  unsigned sample_idx = start_sample;
  bool finished = true;

  typename R::State rng;

  ArrayVec<intersect::Ray, C::L::max_sample_size + 1> rays;

  RenderingEquationState<C::L::max_sample_size> state;

  auto intensity = Eigen::Array3f::Zero().eval();
  while (!finished || sample_idx != end_sample) {
    if (finished) {
      rng = rng_ref.get_generator(sample_idx, location);
      auto initial_sample = initial_ray_sampler(rng);
      rays.resize(0);
      rays.push_back(initial_sample.ray);
      state = RenderingEquationState<C::L::max_sample_size>::initial_state(
          initial_sample);
      finished = false;
      sample_idx++;
    }

    ArrayVec<intersect::IntersectionOp<typename I::InfoType>,
             C::L::max_sample_size + 1>
        intersections;

    for (const auto &ray : rays) {
      intersections.push_back(intersectable.intersect(ray));
    }

    debug_assert_assume(intersections.size() == rays.size());
    debug_assert_assume(rays.size() > 0);

    auto output =
        rendering_equation_iteration(state, rng, intersections, settings, inp);

    switch (output.type()) {
    case IterationOutputType::NextIteration: {
      auto [new_state, new_rays] =
          output.get(TAG(IterationOutputType::NextIteration));
      state = new_state;
      rays = new_rays;
    } break;
    case IterationOutputType::Finished:
      intensity += output.get(TAG(IterationOutputType::Finished));
      finished = true;
      break;
    };
  }

  return intensity;
}
} // namespace integrate
