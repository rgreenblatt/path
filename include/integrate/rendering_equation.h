#pragma once

#include "integrate/rendering_equation_components.h"
#include "integrate/rendering_equation_iteration.h"
#include "integrate/rendering_equation_settings.h"
#include "integrate/rendering_equation_state.h"
#include "lib/assert.h"
#include "work_division/location_info.h"

namespace integrate {
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

  ArrayVec<intersect::Ray, C::L::max_num_samples + 1> rays;

  RenderingEquationState<C::L::max_num_samples> state;

  auto intensity = Eigen::Array3f::Zero().eval();
  while (!finished || sample_idx != end_sample) {
    if (finished) {
      rng = rng_ref.get_generator(sample_idx, location);
      auto initial_sample = initial_ray_sampler(rng);
      rays.resize(0);
      rays.push_back(initial_sample.ray);
      state = RenderingEquationState<C::L::max_num_samples>::initial_state(
          initial_sample);
      finished = false;
      sample_idx++;
    }

    ArrayVec<intersect::IntersectionOp<typename I::InfoType>,
             C::L::max_num_samples + 1>
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
