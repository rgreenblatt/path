#pragma once

#include "integrate/rendering_equation_iteration.h"
#include "kernel/kernel_launch.h"
#include "kernel/location_info.h"
#include "kernel/progress_bar_launch.h"
#include "lib/integer_division_utils.h"
#include "meta/all_values/dispatch.h"
#include "meta/all_values/impl/integral.h"
#include "render/detail/initial_ray_sample.h"
#include "render/detail/integrate_image.h"
#include "render/detail/max_blocks_per_launch.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <ExecutionModel exec>
template <typename... T>
void IntegrateImage<exec>::run(IntegrateImageBulkInputs<exec, T...> inp) {
  const auto &val = inp.val;
  always_assert(val.division.block_size() <= val.intersector.max_size());

  unsigned max_launch_size =
      std::min(val.intersector.max_size() / val.division.block_size(),
               max_blocks_per_launch<exec>(val.settings.computation_settings));

  kernel::progress_bar_launch(
      val.division, max_launch_size, val.show_progress,
      [&](unsigned start, unsigned end) {
        unsigned grid = end - start;

        unsigned initial_size = grid * val.division.block_size();
        auto ray_writer = val.intersector.ray_writer(initial_size);
        val.state.state.resize(initial_size);
        val.state.rng_state.resize(initial_size);

        Span state_span = val.state.state;
        Span rng_state_span = val.state.rng_state;

        // be precise with copies...
        auto get_local_idx = [=](const kernel::WorkDivision &division,
                                 unsigned block_idx, unsigned thread_idx) {
          return (block_idx - start) * division.block_size() + thread_idx;
        };

        auto rng = val.items.rng;
        auto film_to_world = val.items.film_to_world;

        const auto &components = val.items.components;
        constexpr unsigned max_num_light_samples =
            std::decay_t<decltype(components)>::L::max_num_samples;

    // initialize
    // TODO: SPEED: change work division used for this kernel???
#pragma message "fix this - kernel launch"
#if 0
    kernel::KernelLaunch<exec>::run(
        division, start, end,
        [=] HOST_DEVICE(const kernel::WorkDivision &division,
                        const kernel::GridLocationInfo &info,
                        unsigned block_idx, unsigned thread_idx) {
          kernel::LocationInfo loc_info =
              kernel::LocationInfo::from_grid_location_info(info,
                                                            division.x_dim());
          const unsigned local_idx =
              get_local_idx(division, block_idx, thread_idx);
          for (unsigned sample_idx = info.start_sample;
               sample_idx < info.end_sample; ++sample_idx) {
            rng_state_span[local_idx] =
                rng.get_generator(sample_idx, loc_info.location);
            auto initial_sample = initial_ray_sample(
                rng_state_span[local_idx], info.x, info.y, division.x_dim(),
                division.y_dim(), film_to_world);
            ray_writer.write_at(local_idx, initial_sample.ray);
            state_span[local_idx] = integrate::RenderingEquationState<
                max_num_light_samples>::initial_state(initial_sample);
          }
        });
#endif

        bool has_remaining_samples = true;

        bool is_first = true;

        unsigned current_size = initial_size;

        while (has_remaining_samples) {
          auto intersections_span = val.intersector.get_intersections();
          auto rendering_settings = val.settings.rendering_equation_settings;
          auto bgra_32 = val.items.base.bgra_32;
          auto float_rgb = val.items.base.float_rgb;

          Span initial_state_span = val.state.state;
          val.state.op_state.resize(current_size);
          Span op_state_span = val.state.op_state;

          dispatch(is_first, [&](auto tag) {
            constexpr bool is_first = tag;

#pragma message "fix this - kernel launch"
#if 0
        kernel::KernelLaunch<exec>::run(
            division, start, end,
            [=] HOST_DEVICE(const kernel::WorkDivision &division,
                            const kernel::GridLocationInfo &,
                            unsigned block_idx, unsigned thread_idx) {
              bool has_value = true;
              const unsigned local_idx =
                  get_local_idx(division, block_idx, thread_idx);
              if constexpr (!is_first) {
                has_value = op_state_span[local_idx].has_value();
              }
              if (has_value) {
                const auto &previous_state = [&]() -> const auto & {
                  if constexpr (is_first) {
                    return initial_state_span[local_idx];
                  } else {
                    return *op_state_span[local_idx];
                  }
                }
                ();
                auto &next_state = op_state_span[local_idx];

                constexpr unsigned max_samples = max_num_light_samples + 1;

                ArrayVec<std::remove_cv_t<
                             typename decltype(intersections_span)::ValueType>,
                         max_samples>
                    intersections;
                if constexpr (is_first) {
                  intersections.push_back(intersections_span[local_idx]);
                } else {
                  debug_assert_assume(previous_state.num_samples() <=
                                      max_samples);

#pragma unroll max_samples
                  for (unsigned i = 0; i < previous_state.num_samples(); ++i) {
                    intersections.push_back(
                        intersections_span[local_idx * max_samples + i]);
                  }
                }

                auto next = rendering_equation_iteration(
                    state_span[local_idx], rng_state_span[local_idx],
                    intersections, rendering_settings, components);

                next.visit_tagged([&](auto tag, auto &value) {
                  constexpr auto type = tag;
                  if constexpr (type ==
                                integrate::IterationOutputType::NextIteration) {
                    next_state = value.state;
#pragma message "HERE TODO: rays"
                  } else {
                    static_assert(type ==
                                  integrate::IterationOutputType::Finished);
                    next_state = std::nullopt;
#pragma message "HERE TODO"
                  }
                });
                // next.
              }
            });
#endif
          });
          is_first = false;
        }
      });
}
} // namespace detail
} // namespace render
