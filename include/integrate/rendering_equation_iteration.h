#pragma once

#include "integrate/rendering_equation_components.h"
#include "integrate/rendering_equation_settings.h"
#include "integrate/rendering_equation_state.h"
#include "lib/assert.h"

namespace integrate {
template <rng::RngState R, ExactSpecializationOf<RenderingEquationComponents> C>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline IterationOutput<C::L::max_num_samples>
rendering_equation_iteration(
    const RenderingEquationState<C::L::max_num_samples> &state, R &rng,
    const std::optional<intersect::Ray> &last_ray,
    SpanSized<const intersect::IntersectionOp<typename C::InfoType>>
        intersections,
    const RenderingEquationSettings &settings, const C &inp) {
  const auto &[iters, count_emission, has_next_sample, ray_info, light_samples,
               old_float_rgb_total] = state;
  const auto &[scene, light_sampler, dir_sampler, term_prob] = inp;

  auto use_intersection = [&](const auto &intersection,
                              std::optional<float> target_distance) {
    return intersection.has_value() &&
           (!target_distance.has_value() ||
            std::abs(*target_distance - intersection->intersection_dist) <
                1e-6);
  };

  auto include_lighting = [&](const auto &intersection) {
    // TODO it might be a good idea to more generally have a setting
    // to back cull everything except bsdfs
    return !settings.back_cull_emission || !intersection.is_back_intersection;
  };

  RenderingEquationState<C::L::max_num_samples> new_state;
  new_state.iters = iters + 1;
  new_state.float_rgb_total = old_float_rgb_total;
  auto &float_rgb_total = new_state.float_rgb_total;

  debug_assert_assume(light_samples.size() ==
                      intersections.size() - has_next_sample);
  debug_assert(has_next_sample == last_ray.has_value());
  debug_assert_assume(light_samples.size() <= C::L::max_num_samples);
  for (unsigned i = 0; i < light_samples.size(); ++i) {
    const auto &[multiplier, target_distance] = light_samples[i];
    const auto &intersection_op = intersections[i];

    if (!use_intersection(intersection_op, target_distance) ||
        !include_lighting(*intersection_op)) {
      continue;
    }

    float_rgb_total +=
        scene.get_material(*intersection_op).emission * multiplier;
  }

  if (!has_next_sample) {
    return {tag_v<IterationOutputType::Finished>, float_rgb_total};
  }

  debug_assert(intersections.size() > 0);

  const auto &ray = *last_ray;
  const auto &next_intersection_op = intersections[intersections.size() - 1];

  if (!use_intersection(next_intersection_op, ray_info.target_distance)) {
    return {tag_v<IterationOutputType::Finished>, float_rgb_total};
  }

  const auto &next_intersection = *next_intersection_op;
  const auto &intersection_point = next_intersection.intersection_point(ray);
  FloatRGB multiplier = ray_info.multiplier;

  decltype(auto) material = scene.get_material(next_intersection);

  if ((!C::L::performs_samples || count_emission) &&
      include_lighting(next_intersection)) {
    float_rgb_total += multiplier * material.emission;
  }

  decltype(auto) normal = scene.get_normal(next_intersection, ray);

  using B = typename C::B;

  ArrayVec<intersect::Ray, C::L::max_num_samples + 1> new_rays;

  if constexpr (B::continuous) {
    auto add_direct_lighting = [&, &multiplier =
                                       multiplier](float prob_continuous) {
      const auto samples = inp.light_sampler(intersection_point, material,
                                             ray.direction, normal, rng);
      for (const auto &[dir_sample, light_target_distance] : samples) {
        // brdf isn't required to handle this case (and light sampler must
        // filter)
        debug_assert(!material.bsdf.is_brdf() ||
                     dir_sample.direction->dot(*normal) >= -1e-6);

        intersect::Ray light_ray{intersection_point, dir_sample.direction};

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

  FloatRGB new_multiplier = multiplier * sample.sample.multiplier;

  auto this_term_prob = term_prob(iters, new_multiplier);

  new_state.has_next_sample = rng.next() > this_term_prob;

  if (new_state.has_next_sample) {
    new_multiplier /= (1.0f - this_term_prob);

    debug_assert(new_multiplier[0] >= 0.0f);
    debug_assert(new_multiplier[0] >= 0.0f);
    debug_assert(new_multiplier[0] >= 0.0f);

    intersect::Ray new_ray{intersection_point, sample.sample.direction};
    new_state.ray_info = {new_multiplier, std::nullopt};
    new_rays.push_back(new_ray);
  } else {
    if (new_rays.size() == 0) {
      return {tag_v<IterationOutputType::Finished>, float_rgb_total};
    }
  }

  return {tag_v<IterationOutputType::NextIteration>, new_state, new_rays};
}

} // namespace integrate
