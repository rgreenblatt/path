#pragma once

#include "execution_model/thrust_data.h"
#include "integrate/rendering_equation_iteration.h"
#include "kernel/atomic.h"
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

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace render {
namespace detail {
namespace integrate_image_bulk {
namespace detail {
// returns sample count
template <ExecutionModel exec, typename Inp, typename State>
void initialize(const Inp &inp, State &bulk_state, unsigned samples_per_loc,
                unsigned start_block_init, unsigned end_block_init,
                bool parity) {
  auto get_sample = [&](unsigned block_idx) {
    auto thread_info = inp.division.get_thread_info(block_idx, 0).info;
    unsigned x_dim = inp.division.x_dim();
    // full 64 is required to avoid overflow (check!)
    uint64_t locs_before =
        thread_info.y * x_dim + std::min(thread_info.x, x_dim - 1);
    return samples_per_loc * locs_before + thread_info.start_sample;
  };

  auto start_thread_info =
      inp.division.get_thread_info(start_block_init, 0).info;
  unsigned start_x = start_thread_info.x;
  unsigned start_y = start_thread_info.y;
  unsigned start_loc_sample = start_thread_info.start_sample;

  uint64_t start_overall_sample = get_sample(start_block_init);
  uint64_t end_overall_sample = get_sample(end_block_init);
  unsigned sample_count = end_overall_sample - start_overall_sample;

  auto &rays_in = bulk_state.rays_in_and_out[parity];
  auto &rng_states_in = bulk_state.rng_states_in_and_out[parity];

  unsigned existing_rays = rays_in.size();
  unsigned new_rays = existing_rays + sample_count;
  rays_in.resize(new_rays);

  unsigned existing_states = rng_states_in.size();
  unsigned new_states = existing_states + sample_count;
  rng_states_in.resize(new_states);

  Span rays_in_span = Span{rays_in}.slice(existing_rays, new_rays).as_unsized();
  Span rng_states_in_span =
      Span{rng_states_in}.slice(existing_states, new_states).as_unsized();

  bulk_state.initial_sample_info.resize(sample_count);
  Span initial_sample_info = Span{bulk_state.initial_sample_info}.as_unsized();

  auto rng = inp.items.rng;
  auto film_to_world = inp.items.film_to_world;

  // initialize
  // TODO: SPEED: change work division used for this kernel???
  kernel::KernelLaunch<exec>::run(
      inp.division, start_block_init, end_block_init,
      kernel::make_no_interactor_launchable(
          [=] HOST_DEVICE(const kernel::WorkDivision &division,
                          const kernel::GridLocationInfo &info, unsigned,
                          unsigned, const auto &, const auto &) {
            // note that info.x - start_x may underflow (which is fine)
            unsigned locs_from_start =
                (info.y - start_y) * division.x_dim() + (info.x - start_x);

            unsigned loc_local_idx = locs_from_start * samples_per_loc;

            kernel::LocationInfo loc_info =
                kernel::LocationInfo::from_grid_location_info(info,
                                                              division.x_dim());
            for (unsigned sample_idx = info.start_sample;
                 sample_idx < info.end_sample; ++sample_idx) {
              unsigned local_idx =
                  loc_local_idx + (sample_idx - start_loc_sample);
              rng_states_in_span[local_idx] =
                  rng.get_generator(sample_idx, loc_info.location);
              auto [ray, sample_info] = initial_ray_sample(
                  rng_states_in_span[local_idx], info.x, info.y,
                  division.x_dim(), division.y_dim(), film_to_world);
              rays_in_span[local_idx] = ray;
              initial_sample_info[local_idx] = sample_info;
            }
          }));
}

void shade() {}
} // namespace detail
} // namespace integrate_image_bulk

struct ShadingWorkDivision {
  unsigned block_size;
  unsigned samples_per_thread;
};

// TODO: inp.division is just for init???
// TODO: inp.settings.computation_settings is just for init???
template <ExecutionModel exec>
template <ExactSpecializationOf<IntegrateImageItems> Items,
          intersectable_scene::BulkIntersector I>
requires std::same_as<typename Items::InfoType, typename I::InfoType>
void IntegrateImageBulk<exec>::run(
    IntegrateImageInputs<Items> inp, I &intersector,
    IntegrateImageBulkState<exec, Items::C::max_num_light_samples(),
                            typename Items::R> &bulk_state,
    const unsigned min_additional_samples) {
  using namespace integrate_image_bulk::detail;

  // unimplemented!
  unreachable();
  always_assert(inp.division.block_size() <= intersector.max_size());

  constexpr unsigned max_num_light_samples_per =
      Items::C::max_num_light_samples();

  // add 1 for next bounce
  constexpr unsigned max_rays_per_sample = max_num_light_samples_per + 1;

  unsigned max_num_extra_samples_per_thread =
      inp.division.n_threads_per_unit_extra() > 0 ? 1 : 0;
  unsigned max_samples_per_thread =
      inp.division.base_samples_per_thread() + max_num_extra_samples_per_thread;

  bool has_multiple_sample_blocks = inp.division.num_sample_blocks() > 1;

  // TODO: consider extracting back to division
  unsigned samples_per_loc = inp.division.base_samples_per_thread() +
                             inp.division.n_threads_per_unit_extra();

  // In the case where we have samples over multiple different blocks, we
  // can upperbound with max_samples_per_thread.
  // Otherwise, we can just use samples per.
  unsigned max_samples_per_loc =
      has_multiple_sample_blocks
          ? inp.division.sample_block_size() * max_samples_per_thread
          : samples_per_loc;

  unsigned num_locs = inp.division.x_block_size() * inp.division.y_block_size();

  unsigned max_samples_per_init_block = num_locs * max_samples_per_loc;

  unsigned max_num_rays = intersector.max_size();

  // add these as input
  unsigned max_num_samples_per_launch_input_TODO = 0;
  unsigned min_init_launch_input_TODO = 0;

  unsigned max_num_samples_per_launch =
      std::min(max_num_rays / max_rays_per_sample,
               max_num_samples_per_launch_input_TODO);

  unsigned total_num_init_blocks = inp.division.total_num_blocks();

  unsigned start_block_init = 0;

  ProgressBar progress_bar(total_num_init_blocks, 70);
  if (inp.show_progress) {
    progress_bar.display();
  }

  bool parity = false;
  bulk_state.rays_in_and_out[parity].resize(0);
  bulk_state.rays_in_and_out[!parity].resize(0);
  bulk_state.states_in_and_out[parity].resize(0);
  bulk_state.states_in_and_out[!parity].resize(0);
  bulk_state.ray_idx.set(0);
  bulk_state.state_idx.set(0);

  unsigned num_states = 0;

  while (start_block_init != total_num_init_blocks || num_states != 0) {
    // reset back to zero for launches
    bulk_state.state_idx.set(0);

    unsigned max_init_blocks_per_launch =
        max_blocks_per_launch<exec>(inp.settings.computation_settings);

    unsigned max_num_samples_to_init = max_num_samples_per_launch - num_states;
    unsigned init_num_blocks =
        std::min(std::min(max_num_samples_to_init / max_samples_per_init_block,

                          total_num_init_blocks - start_block_init),
                 max_init_blocks_per_launch);

    unsigned attempt_end_block_init = init_num_blocks + start_block_init;

    auto &states_in = bulk_state.states_in_and_out[parity];
    auto &states_out = bulk_state.states_in_and_out[!parity];
    auto &rng_states_in = bulk_state.rng_states_in_and_out[parity];
    auto &rng_states_out = bulk_state.rng_states_in_and_out[!parity];
    auto &rays_in = bulk_state.rays_in_and_out[parity];
    auto &rays_out = bulk_state.rays_in_and_out[!parity];

    unsigned pre_init_ray_size = rays_in.size();
    unsigned pre_init_state_size = rng_states_in.size();

    unsigned end_block_init = 0;
    if (init_num_blocks >= min_init_launch_input_TODO ||
        attempt_end_block_init == total_num_init_blocks) {
      initialize<exec>(inp, bulk_state, samples_per_loc, start_block_init,
                       attempt_end_block_init, parity);
      always_assert(bulk_state.initial_sample_info.size() > 0);
      end_block_init = attempt_end_block_init;
    }

    auto intersections_span = intersector.get_intersections(Span{rays_in});

    states_out.resize(rng_states_in.size());

    using StateT = State<max_num_light_samples_per>;

    Span state_idx_span = bulk_state.state_idx.span().as_unsized();
    Span ray_idx_span = bulk_state.ray_idx.span().as_unsized();

    auto shade = [&]<typename T>(auto tag, SpanSized<const T> states_span_in) {
      constexpr bool is_init = tag;

      auto start = thrust::make_counting_iterator(0u);
      auto end = start + states_span_in.size();

      auto states_span = states_span_in.as_unsized();

      Span states_out_span = Span{states_out}.as_unsized();
      Span rng_states_out_span = Span{rng_states_out}.as_unsized();
      Span rays_out_span = Span{rays_out}.as_unsized();
      Span rng_states_in_span = Span{rng_states_in}.as_unsized().as_const();
      Span rays_in_span = Span{rays_in}.as_unsized().as_const();
      auto rendering_equation_settings =
          inp.settings.rendering_equation_settings;
      auto components = inp.items.components;

      thrust::for_each(
          ThrustData<exec>().execution_policy(), start, end,
          [=] HOST_DEVICE(unsigned idx) {
            const auto &inp_state = states_span[idx];
            unsigned overall_state_idx = [=]() {
              if constexpr (is_init) {
                return idx + pre_init_state_size;

              } else {
                return idx;
              }
            }();
            decltype(auto) state = [&]() -> decltype(auto) {
              if constexpr (is_init) {
                return StateT{
                    .render_state = integrate::RenderingEquationState<
                        max_num_light_samples_per>::initial_state(inp_state),
                    .ray_idx = idx + pre_init_ray_size,
                };
              } else {
                return inp_state;
              }
            }();

            unsigned end_ray = state.ray_idx + state.render_state.num_samples();

            auto ray_span = rays_in_span.slice(state.ray_idx, end_ray);
            auto last_ray = state.render_state.get_last_ray(ray_span);
            auto intersections =
                intersections_span.slice(state.ray_idx, end_ray);
            auto rng_state = rng_states_in_span[overall_state_idx];
            auto next = rendering_equation_iteration(
                state.render_state, rng_state, last_ray, intersections,
                rendering_equation_settings, components);

            next.visit_tagged([&](auto tag, auto &value) {
              constexpr auto type = tag;
              if constexpr (type ==
                            integrate::IterationOutputType::NextIteration) {
                // standard atomic idx fetch add approach (like in filter)...
                const unsigned new_state_idx = state_idx_span[0].fetch_add(1);
                const unsigned new_ray_idx =
                    ray_idx_span[0].fetch_add(value.rays.size());
                states_out_span[new_state_idx] = {.render_state = value.state,
                                                  .ray_idx = new_ray_idx};
                rng_states_out_span[new_state_idx] = rng_state;
                debug_assert_assume(value.rays.size() <= max_rays_per_sample);
#pragma unroll max_rays_per_sample
                for (unsigned i = 0; i < value.rays.size(); ++i) {
                  rays_out_span[i + new_ray_idx] = value.rays[i];
                }
              } else {
                static_assert(type == integrate::IterationOutputType::Finished);
              }
            });
          });
    };

    always_assert(rng_states_in.size() ==
                  (states_in.size() + bulk_state.initial_sample_info.size()));

    unsigned max_out_states = rng_states_in.size();
    rays_out.resize(max_out_states * max_rays_per_sample);
    states_out.resize(max_out_states);
    rng_states_out.resize(max_out_states);

    shade(tag_v<false>, Span{states_in}.as_const());
    shade(tag_v<true>, Span{bulk_state.initial_sample_info}.as_const());

    rays_out.resize(bulk_state.ray_idx.get().as_inner());

    num_states = bulk_state.state_idx.get().as_inner();
    always_assert(num_states <= max_num_samples_per_launch);

    states_out.resize(num_states);
    rng_states_out.resize(num_states);

    start_block_init = end_block_init;
    parity = !parity;

    if (inp.show_progress) {
      progress_bar += end_block_init - start_block_init;
      progress_bar.display();
    }
  }

  if (inp.show_progress) {
    progress_bar.done();
  }
}
} // namespace detail
} // namespace render
