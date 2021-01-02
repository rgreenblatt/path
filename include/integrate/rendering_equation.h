#pragma once

#include "integrate/dir_sample.h"
#include "integrate/dir_sampler/dir_sampler.h"
#include "integrate/light_sampler/light_sampler.h"
#include "integrate/rendering_equation_settings.h"
#include "integrate/term_prob/term_prob.h"
#include "intersectable_scene/intersectable_scene.h"
#include "lib/assert.h"
#include "lib/tagged_union.h"

namespace integrate {
namespace detail {
template <typename T> struct RayInfo {
  T multiplier;
  Optional<float> target_distance;
};

template <typename T> struct RayRayInfo {
  intersect::Ray ray;
  RayInfo<T> ray_info;
};
} // namespace detail

using FRayInfo = detail::RayInfo<float>;
using ArrRayInfo = detail::RayInfo<Eigen::Array3f>;
using FRayRayInfo = detail::RayRayInfo<float>;
using ArrRayRayInfo = detail::RayRayInfo<Eigen::Array3f>;

// this should probably be a class with a friend struct...
template <unsigned n_light_samples> struct RenderingEquationState {
  HOST_DEVICE static RenderingEquationState
  initial_state(const FRayInfo &ray_info) {
    return RenderingEquationState{
        .iters = 0,
        .count_emission = true,
        .has_next_sample = true,
        .ray_info = {Eigen::Array3f::Constant(ray_info.multiplier),
                     ray_info.target_distance},
        .light_samples = {},
        .intensity = Eigen::Array3f::Zero(),
    };
  }

  unsigned iters;
  bool count_emission;
  bool has_next_sample;
  ArrRayInfo ray_info;
  ArrayVec<ArrRayInfo, n_light_samples> light_samples;
  Eigen::Array3f intensity;
};

template <unsigned n_light_samples> struct RenderingEquationNextIteration {
  RenderingEquationState<n_light_samples> state;
  ArrayVec<intersect::Ray, n_light_samples + 1> rays;
};

enum class IterationOutputType {
  NextIteration,
  Finished,
};

template <unsigned n_light_samples>
using IterationOutput =
    TaggedUnion<IterationOutputType,
                RenderingEquationNextIteration<n_light_samples>,
                Eigen::Array3f>;

// TODO: initial_ray_sampler concept
template <typename InfoType, intersectable_scene::SceneRef<InfoType> S,
          light_sampler::LightSamplerRef<typename S::B> L,
          dir_sampler::DirSamplerRef<typename S::B> D, term_prob::TermProbRef T,
          rng::RngState R>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline IterationOutput<L::max_sample_size>
rendering_equation_iteration(
    const RenderingEquationState<L::max_sample_size> &state, R &rng,
    const ArrayVec<intersect::IntersectionOp<InfoType>, L::max_sample_size + 1>
        &intersections,
    const intersect::Ray &last_ray, const RenderingEquationSettings &settings,
    const S &scene, const L &light_sampler, const D &dir_sampler,
    const T &term_prob) {
  const auto &[iters, count_emission, has_next_sample, ray_info, light_samples,
               old_intensity] = state;

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

  auto intensity = old_intensity;

  debug_assert_assume(light_samples.size() ==
                      intersections.size() - has_next_sample);
  debug_assert_assume(light_samples.size() <= L::max_sample_size);
  for (unsigned i = 0; i < light_samples.size(); ++i) {
    const auto &[multiplier, target_distance] = light_samples[i];
    const auto &intersection_op = intersections[i];

    if (!use_intersection(intersection_op, target_distance) ||
        !include_lighting(*intersection_op)) {
      continue;
    }

    intensity += scene.get_material(*intersection_op).emission * multiplier;
  }

  auto finish = [&] {
    return IterationOutput<L::max_sample_size>::template create<
        IterationOutputType::Finished>(intensity);
  };

  if (!has_next_sample) {
    return finish();
  }

  debug_assert(intersections.size() > 0);

  const auto &[multiplier, target_distance] = ray_info;
  const auto &next_intersection_op = intersections[intersections.size() - 1];

  if (!use_intersection(next_intersection_op, target_distance)) {
    return finish();
  }

  const auto &next_intersection = *next_intersection_op;
  const auto &ray = last_ray;
  const auto &intersection_point = next_intersection.intersection_point(ray);

  // FIXME references...
  const auto &material = scene.get_material(next_intersection);

  if ((!L::performs_samples || count_emission) &&
      include_lighting(next_intersection)) {
    intensity += multiplier * material.emission;
  }

  const auto &&normal = scene.get_normal(next_intersection, last_ray);

  RenderingEquationState<L::max_sample_size> new_state;
  new_state.iters = iters + 1;

  using B = typename S::B;

  ArrayVec<intersect::Ray, L::max_sample_size + 1> new_rays;

  if constexpr (B::continuous) {
    auto add_direct_lighting = [&, &multiplier =
                                       multiplier](float prob_continuous) {
      const auto samples = light_sampler(intersection_point, material,
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
            multiplier *
            material.bsdf.continuous_eval(ray.direction, light_ray.direction,
                                          normal) *
            prob_continuous / dir_sample.multiplier;

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

    new_state.ray_info = {new_multiplier, nullopt_value};
    new_rays.push_back({intersection_point, sample.sample.direction});
  } else {
    if (new_rays.size() == 0) {
      return finish();
    }
  }

  return IterationOutput<L::max_sample_size>::template create<
      IterationOutputType::NextIteration>(new_state, new_rays);
}

template <typename F, intersect::Intersectable I,
          intersectable_scene::SceneRef<typename I::InfoType> S,
          light_sampler::LightSamplerRef<typename S::B> L,
          dir_sampler::DirSamplerRef<typename S::B> D, term_prob::TermProbRef T,
          rng::RngRef R>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline Eigen::Array3f rendering_equation(
    const F &initial_ray_sampler, unsigned start_sample, unsigned end_sample,
    unsigned location, const RenderingEquationSettings &settings,
    const I &intersectable, const S &scene, const L &light_sampler,
    const D &dir_sampler, const T &term_prob, const R &rng_ref) {
  unsigned sample_idx = start_sample;
  bool finished = true;

  typename R::State rng;

  ArrayVec<intersect::Ray, L::max_sample_size + 1> rays;

  RenderingEquationState<L::max_sample_size> state;

  auto intensity = Eigen::Array3f::Zero().eval();
  while (!finished || sample_idx != end_sample) {
    if (finished) {
      rng = rng_ref.get_generator(sample_idx, location);
      auto [ray_v, sample] = initial_ray_sampler(rng);
      rays.push_back(ray_v);
      state = RenderingEquationState<L::max_sample_size>::initial_state(sample);
      finished = false;
      sample_idx++;
    }

    ArrayVec<intersect::IntersectionOp<typename I::InfoType>,
             L::max_sample_size + 1>
        intersections;

    for (const auto &ray : rays) {
      intersections.push_back(intersectable.intersect(ray));
    }

    debug_assert_assume(intersections.size() == rays.size());
    debug_assert_assume(rays.size() > 0);

    auto output = rendering_equation_iteration(
        state, rng, intersections, rays[rays.size() - 1], settings, scene,
        light_sampler, dir_sampler, term_prob);

    switch (output.type()) {
    case IterationOutputType::NextIteration: {
      auto [new_state, new_rays] =
          output.template get<IterationOutputType::NextIteration>();
      state = new_state;
      rays = new_rays;
    } break;
    case IterationOutputType::Finished:
      intensity += output.template get<IterationOutputType::Finished>();
      finished = true;
      break;
    };
  }

  return intensity;
}
} // namespace integrate
