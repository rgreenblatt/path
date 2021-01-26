#pragma once

#include "integrate/rendering_equation_iteration.h"
#include "kernel/kernel_launch.h"
#include "kernel/location_info.h"
#include "kernel/make_no_interactor_launchable.h"
#include "kernel/progress_bar_launch.h"
#include "lib/integer_division_utils.h"
#include "meta/all_values/dispatch.h"
#include "meta/all_values/impl/integral.h"
#include "render/detail/initial_ray_sample.h"
#include "render/detail/integrate_image_bulk.h"
#include "render/detail/max_blocks_per_launch.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <ExecutionModel exec>
template <ExactSpecializationOf<IntegrateImageItems> Items,
          intersectable_scene::BulkIntersector I>
requires std::same_as<typename Items::InfoType, typename I::InfoType>
void IntegrateImageBulk<exec>::run(
    IntegrateImageInputs<Items> inp, I &intersector,
    IntegrateImageBulkState<exec, Items::C::L::max_num_samples,
                            typename Items::R> &state) {
  // unimplemented!
  unreachable();
  always_assert(inp.division.block_size() <= intersector.max_size());

  unsigned max_launch_size =
      std::min(intersector.max_size() / inp.division.block_size(),
               max_blocks_per_launch<exec>(inp.settings.computation_settings));

  kernel::progress_bar_launch(
      inp.division, max_launch_size, inp.show_progress,
      [&](unsigned start, unsigned end) {
        unsigned grid = end - start;

        unsigned initial_size = grid * inp.division.block_size();

        state.rng_state.resize(initial_size);
        state.initial_sample_info.resize(initial_size);
        state.rays.resize(initial_size);

        Span rng_state = state.rng_state;
        Span initial_sample_info = state.initial_sample_info;
        Span rays = state.rays;

        // be precise with copies...
        auto get_local_idx = [=](const kernel::WorkDivision &division,
                                 unsigned block_idx, unsigned thread_idx) {
          return (block_idx - start) * division.block_size() + thread_idx;
        };

        auto rng = inp.items.rng;
        auto film_to_world = inp.items.film_to_world;

        const auto &components = inp.items.components;
        // initialize
        // TODO: SPEED: change work division used for this kernel???
        kernel::KernelLaunch<exec>::run(
            inp.division, start, end,
            kernel::make_no_interactor_launchable(
                [=] HOST_DEVICE(const kernel::WorkDivision &division,
                                const kernel::GridLocationInfo &info,
                                unsigned block_idx, unsigned thread_idx,
                                const auto &, const auto &) {
                  kernel::LocationInfo loc_info =
                      kernel::LocationInfo::from_grid_location_info(
                          info, division.x_dim());
                  const unsigned local_idx =
                      get_local_idx(division, block_idx, thread_idx);
                  for (unsigned sample_idx = info.start_sample;
                       sample_idx < info.end_sample; ++sample_idx) {
                    rng_state[local_idx] =
                        rng.get_generator(sample_idx, loc_info.location);
                    auto [ray, sample_info] = initial_ray_sample(
                        rng_state[local_idx], info.x, info.y, division.x_dim(),
                        division.y_dim(), film_to_world);
                    rays[local_idx] = ray;
                    initial_sample_info[local_idx] = sample_info;
                  }
                }));

        constexpr unsigned max_num_light_samples =
            std::decay_t<decltype(components)>::L::max_num_samples;

        // state_span[local_idx] = integrate::RenderingEquationState<
        //     max_num_light_samples>::initial_state(initial_sample_info);

        bool has_remaining_samples = true;

        bool is_first = true;

        unsigned current_size = initial_size;

        while (has_remaining_samples) {
          auto rendering_settings = inp.settings.rendering_equation_settings;
          auto bgra_32 = inp.items.base.bgra_32;
          auto float_rgb = inp.items.base.float_rgb;

          state.op_state.resize(current_size);
          state.sample_rays.resize(current_size);
          Span op_state_span = state.op_state;
          Span sample_rays = state.sample_rays;

          dispatch(is_first, [&](auto tag) {
            constexpr bool is_first = tag;

            auto intersections_span = [&] {
              if constexpr (is_first) {
                return intersector.get_intersections(rays);
              } else {
                return intersector.get_intersections(sample_rays);
              }
            }();

            kernel::KernelLaunch<exec>::run(
                inp.division, start, end,
                // TODO: will need interactor!
                kernel::make_no_interactor_launchable(
                    [=] HOST_DEVICE(const kernel::WorkDivision &division,
                                    const kernel::GridLocationInfo &,
                                    unsigned block_idx, unsigned thread_idx,
                                    const auto &, const auto &) {
                      bool has_value = true;
                      const unsigned local_idx =
                          get_local_idx(division, block_idx, thread_idx);
                      if constexpr (!is_first) {
                        has_value = op_state_span[local_idx].has_value();
                      }
                      if (has_value) {
                        // TODO: this will need to change if atomics/etc are
                        // used here
                        auto previous_state = [&]() {
                          if constexpr (is_first) {
                            return integrate::RenderingEquationState<
                                max_num_light_samples>::
                                initial_state(initial_sample_info[local_idx]);
                          } else {
                            return *op_state_span[local_idx];
                          }
                        }();
                        auto previous_ray = [&]() {
                          if constexpr (is_first) {
                            return std::make_optional(rays[local_idx]);
                          } else {
                            return previous_state.get_last_ray(
                                sample_rays[local_idx]);
                          }
                        }();
                        auto &next_state = op_state_span[local_idx];

                        constexpr unsigned max_samples =
                            max_num_light_samples + 1;

                        ArrayVec<std::remove_cv_t<
                                     typename decltype(intersections_span)::
                                         ValueType>,
                                 max_samples>
                            intersections;
                        if constexpr (is_first) {
                          intersections.push_back(
                              intersections_span[local_idx]);
                        } else {
                          debug_assert_assume(previous_state.num_samples() <=
                                              max_samples);

#pragma unroll max_samples
                          for (unsigned i = 0; i < previous_state.num_samples();
                               ++i) {
                            intersections.push_back(
                                intersections_span[local_idx * max_samples +
                                                   i]);
                          }
                        }

                        auto next = rendering_equation_iteration(
                            previous_state, rng_state[local_idx], previous_ray,
                            intersections, rendering_settings, components);

                        next.visit_tagged([&](auto tag, auto &value) {
                          constexpr auto type = tag;
                          if constexpr (type == integrate::IterationOutputType::
                                                    NextIteration) {
                            next_state = value.state;
#pragma message "HERE TODO: rays"
                          } else {
                            static_assert(
                                type ==
                                integrate::IterationOutputType::Finished);
                            next_state = std::nullopt;
#pragma message "HERE TODO (update as needed)"
                          }
                        });
                      }
                    }));
          });
          is_first = false;
          // combine/reduce as needed
        }
      });
}
} // namespace detail
} // namespace render
