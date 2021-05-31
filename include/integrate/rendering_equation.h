#pragma once

#include "integrate/rendering_equation_components.h"
#include "integrate/rendering_equation_iteration.h"
#include "integrate/rendering_equation_settings.h"
#include "integrate/rendering_equation_state.h"
#include "kernel/location_info.h"
#include "lib/assert.h"
#include "lib/float_rgb.h"

namespace integrate {
template <typename T>
concept InitialRaySampler = requires(const T &v, rng::MockRngState &rng) {
  { v(rng) } -> std::same_as<FRayRayInfo>;
};

template <typename T, typename InfoType>
concept InitialIntersectionSampler = requires(const T &v,
                                              rng::MockRngState &rng) {
  { v(rng) } -> std::same_as<IntersectionInfo<InfoType>>;
};

template <typename T, typename InfoType>
concept Sampler =
    InitialRaySampler<T> || InitialIntersectionSampler<T, InfoType>;

template <bool output_per_step, rng::RngRef R, intersect::Intersectable I,
          Sampler<typename I::InfoType> F,
          ExactSpecializationOf<RenderingEquationComponents> C>
ATTR_NO_DISCARD_PURE
    HOST_DEVICE inline std::conditional_t<output_per_step, void, FloatRGB>
    rendering_equation(
        const kernel::LocationInfo &location_info,
        const RenderingEquationSettings &settings, const F &initial_sampler,
        const R &rng_ref, const I &intersectable, const C &inp,
        std::conditional_t<output_per_step, SpanSized<FloatRGB>, std::tuple<>>
            step_outputs) {
  const auto &[start_sample, end_sample, location] = location_info;

  unsigned sample_idx = start_sample;
  bool finished = true;
  unsigned current_step = 0;

  typename R::State rng;

  ArrayVec<intersect::Ray, C::L::max_num_samples + 1> rays;

  RenderingEquationState<C::L::max_num_samples> state;

  FloatRGB float_rgb_total = FloatRGB::Zero();
  while (!finished || sample_idx != end_sample) {
    ArrayVec<intersect::IntersectionOp<typename I::InfoType>,
             C::L::max_num_samples + 1>
        intersections;

    if (finished) {
      rays.resize(0);
      rng = rng_ref.get_generator(sample_idx, location);
      auto initial_sample = initial_sampler(rng);

      if constexpr (output_per_step) {
        current_step = 0;
      }

      if constexpr (InitialIntersectionSampler<F, typename I::InfoType>) {
        intersections.push_back(initial_sample.intersection);
        rays.push_back(initial_sample.info.ray);
        state = RenderingEquationState<C::L::max_num_samples>::initial_state(
            initial_sample.info.info);
      } else {
        static_assert(InitialRaySampler<F>);
        rays.push_back(initial_sample.ray);
        state = RenderingEquationState<C::L::max_num_samples>::initial_state(
            initial_sample.info);
      }

      finished = false;
      sample_idx++;
    }

    if constexpr (output_per_step) {
      if (current_step >= step_outputs.size()) {
        finished = true;
        continue;
      }
    }

    if (intersections.empty()) {
      for (const auto &ray : rays) {
        intersections.push_back(intersectable.intersect(ray));
      }
    }

    debug_assert_assume(intersections.size() == rays.size());
    debug_assert_assume(rays.size() > 0);

    auto output = rendering_equation_iteration(
        state, rng, state.get_last_ray(rays), intersections, settings, inp);

    output.visit_tagged([&](auto tag, const auto &value) {
      if constexpr (tag == IterationOutputType::NextIteration) {
        state = value.state;
        rays = value.rays;

        if constexpr (output_per_step) {
          step_outputs[current_step] += state.float_rgb_total;
          state.float_rgb_total = FloatRGB::Zero();
        }
      } else {
        static_assert(tag == IterationOutputType::Finished);

        if constexpr (output_per_step) {
          step_outputs[current_step] += value;
        } else {
          float_rgb_total += value;
        }
        finished = true;
      }
    });

    if constexpr (output_per_step) {
      ++current_step;
    }
  }

  if constexpr (!output_per_step) {
    return float_rgb_total;
  }
}
} // namespace integrate
