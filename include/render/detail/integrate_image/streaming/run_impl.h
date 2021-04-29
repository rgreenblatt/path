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
#include "render/detail/integrate_image/assign_output.h"
#include "render/detail/integrate_image/initial_ray_sample.h"
#include "render/detail/integrate_image/streaming/run.h"

#include <cli/ProgressBar.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace render {
namespace detail {
namespace integrate_image {
namespace streaming {
namespace detail {
// returns sample count
template <ExecutionModel exec, typename Inp, typename State>
void initialize(const Inp &inp, State &state,
                const kernel::WorkDivision &division, unsigned samples_per_loc,
                unsigned start_block_init, unsigned end_block_init,
                bool parity) {
  auto get_sample = [&](unsigned block_idx) {
    auto thread_info = division.get_thread_info(block_idx, 0).info;
    unsigned x_dim = division.x_dim();
    // full 64 is required to avoid overflow (check!)
    uint64_t locs_before =
        thread_info.y * x_dim + std::min(thread_info.x, x_dim - 1);
    return samples_per_loc * locs_before + thread_info.start_sample;
  };

  auto start_thread_info = division.get_thread_info(start_block_init, 0).info;
  unsigned start_x = start_thread_info.x;
  unsigned start_y = start_thread_info.y;
  unsigned start_loc_sample = start_thread_info.start_sample;

  uint64_t start_overall_sample = get_sample(start_block_init);
  uint64_t end_overall_sample = get_sample(end_block_init);
  unsigned sample_count = end_overall_sample - start_overall_sample;

  auto &rays_in = state.rays_in_and_out[parity];
  auto &rng_states_in = state.rng_states_in_and_out[parity];

  unsigned existing_rays = rays_in.size();
  unsigned new_rays = existing_rays + sample_count;
  rays_in.resize(new_rays);

  unsigned existing_states = rng_states_in.size();
  unsigned new_states = existing_states + sample_count;
  rng_states_in.resize(new_states);

  Span rays_in_span = Span{rays_in}.slice(existing_rays, new_rays).as_unsized();
  Span rng_states_in_span =
      Span{rng_states_in}.slice(existing_states, new_states).as_unsized();

  state.initial_sample_info.resize(sample_count);
  Span initial_sample_info = Span{state.initial_sample_info}.as_unsized();

  auto rng = inp.items.rng;
  auto film_to_world = inp.items.film_to_world;

  // initialize
  // TODO: SPEED: change work division used for this kernel???
  kernel::KernelLaunch<exec>::run(
      division, start_block_init, end_block_init,
      kernel::make_no_interactor_launchable(
          [=] HOST_DEVICE(const kernel::WorkDivision &division,
                          const kernel::GridLocationInfo &info, unsigned,
                          unsigned, const auto &, const auto &) {
            // note that info.x - start_x may underflow (which is fine)
            unsigned locs_from_start =
                (info.y - start_y) * division.x_dim() + (info.x - start_x);

            unsigned loc_local_idx = locs_from_start * samples_per_loc;

            unsigned location =
                kernel::get_location(info.x, info.y, division.x_dim());
            for (unsigned sample_idx = info.start_sample;
                 sample_idx < info.end_sample; ++sample_idx) {
              unsigned local_idx =
                  loc_local_idx + (sample_idx - start_loc_sample);
              auto rng_state = rng.get_generator(sample_idx, location);
              auto [ray, ray_info] = initial_ray_sample(
                  rng_state, info.x, info.y, division.x_dim(), division.y_dim(),
                  film_to_world);
              rng_states_in_span[local_idx] = rng_state.save();
              rays_in_span[local_idx] = ray;
              initial_sample_info[local_idx] = {
                  .ray_info = ray_info,
                  .sample_idx = sample_idx,
                  .x = info.x,
                  .y = info.y,
              };
            }
          }));
}
} // namespace detail

template <ExecutionModel exec>
template <ExactSpecializationOf<Items> Items,
          intersectable_scene::BulkIntersector I>
requires std::same_as<typename Items::InfoType, typename I::InfoType>
void Run<exec>::run(
    Inputs<Items> inp, I &intersector,
    State<exec, Items::C::max_num_light_samples(), typename Items::R> &state,
    const StreamingSettings &settings,
    ExecVector<exec, FloatRGB> &float_rgb_out) {
  using namespace detail;

  const kernel::WorkDivision init_division = {
      settings.computation_settings.init_samples_division,
      inp.samples_per,
      inp.x_dim,
      inp.y_dim,
  };

  constexpr unsigned max_num_light_samples_per =
      Items::C::max_num_light_samples();

  // add 1 for next bounce
  constexpr unsigned max_rays_per_sample = max_num_light_samples_per + 1;

  unsigned max_num_extra_samples_per_thread =
      init_division.n_threads_per_unit_extra() > 0 ? 1 : 0;
  unsigned max_samples_per_thread = init_division.base_samples_per_thread() +
                                    max_num_extra_samples_per_thread;

  bool has_multiple_sample_blocks = init_division.num_sample_blocks() > 1;

  // TODO: remove this assert (should be well covered???)
  always_assert(init_division.base_samples_per_thread() *
                        init_division.sample_block_size() +
                    init_division.n_threads_per_unit_extra() ==
                inp.samples_per);

  // In the case where we have samples over multiple different blocks, we
  // can upperbound with max_samples_per_thread.
  // Otherwise, we can just use samples per.
  unsigned max_samples_per_loc =
      has_multiple_sample_blocks
          ? init_division.sample_block_size() * max_samples_per_thread
          : inp.samples_per;

  unsigned num_locs =
      init_division.x_block_size() * init_division.y_block_size();

  unsigned max_samples_per_init_block = num_locs * max_samples_per_loc;

  unsigned max_num_rays = intersector.max_size();

  unsigned max_num_samples_per_launch =
      std::min(max_num_rays / max_rays_per_sample,
               settings.computation_settings.max_num_samples_per_launch);

  unsigned total_num_init_blocks = init_division.total_num_blocks();

  unsigned start_block_init = 0;

  ProgressBar progress_bar(total_num_init_blocks, 70);
  if (inp.show_progress) {
    progress_bar.display();
  }

  bool parity = false;
  state.rays_in_and_out[parity].resize(0);
  state.rays_in_and_out[!parity].resize(0);
  state.states_in_and_out[parity].resize(0);
  state.states_in_and_out[!parity].resize(0);

  state.float_rgb.resize(inp.x_dim * inp.y_dim);
  Span float_rgb_atomic = Span{state.float_rgb}.as_unsized();

  unsigned num_states = 0;

  while (start_block_init != total_num_init_blocks || num_states != 0) {
    // reset back to zero for launches
    state.ray_idx.set(0);
    state.state_idx.set(0);

    unsigned max_num_samples_to_init = max_num_samples_per_launch - num_states;
    unsigned init_num_blocks =
        std::min(max_num_samples_to_init / max_samples_per_init_block,
                 total_num_init_blocks - start_block_init);

    unsigned attempt_end_block_init = init_num_blocks + start_block_init;

    auto &states_in = state.states_in_and_out[parity];
    auto &states_out = state.states_in_and_out[!parity];
    auto &rng_states_in = state.rng_states_in_and_out[parity];
    auto &rng_states_out = state.rng_states_in_and_out[!parity];
    auto &rays_in = state.rays_in_and_out[parity];
    auto &rays_out = state.rays_in_and_out[!parity];
    state.initial_sample_info.resize(0);

    unsigned pre_init_ray_size = rays_in.size();
    unsigned pre_init_state_size = rng_states_in.size();

    unsigned end_block_init = start_block_init;
    if (init_num_blocks >= settings.computation_settings.min_num_init_blocks ||
        (attempt_end_block_init == total_num_init_blocks &&
         init_num_blocks > 0)) {
      initialize<exec>(inp, state, init_division, inp.samples_per,
                       start_block_init, attempt_end_block_init, parity);
      always_assert(state.initial_sample_info.size() > 0);
      end_block_init = attempt_end_block_init;
    }

    auto intersections_span = intersector.get_intersections(Span{rays_in});

    states_out.resize(rng_states_in.size());

    using SampleStateT = SampleState<max_num_light_samples_per>;

    auto shade = [&]<typename T>(auto tag,
                                 SpanSized<const T> sample_states_span_in) {
      constexpr bool is_init = tag;

      auto start = thrust::make_counting_iterator(0u);
      auto end = start + sample_states_span_in.size();

      auto sample_states_span = sample_states_span_in.as_unsized();

      Span states_out_span = Span{states_out}.as_unsized();
      Span rng_states_out_span = Span{rng_states_out}.as_unsized();
      Span rays_out_span = Span{rays_out}.as_unsized();
      Span rng_states_in_span = Span{rng_states_in}.as_unsized().as_const();
      Span rays_in_span = Span{rays_in}.as_unsized().as_const();
      Span state_idx_span = state.state_idx.span().as_unsized();
      Span ray_idx_span = state.ray_idx.span().as_unsized();
      auto components = inp.items.components;
      auto rng = inp.items.rng;

      unsigned x_dim = inp.x_dim;
      auto render_settings = inp.items.render_settings;

      // NOTE: this is very inefficient in the cpu context
      // (the mega kernel approach is much better there anyway)
      thrust::for_each(
          ThrustData<exec>().execution_policy(), start, end,
          [=] HOST_DEVICE(unsigned idx) {
            const auto &inp_state = sample_states_span[idx];
            unsigned overall_state_idx = [=]() {
              if constexpr (is_init) {
                return idx + pre_init_state_size;

              } else {
                return idx;
              }
            }();
            decltype(auto) sample_state = [&]() -> decltype(auto) {
              if constexpr (is_init) {
                return SampleStateT{
                    .render_state = integrate::RenderingEquationState<
                        max_num_light_samples_per>::
                        initial_state(inp_state.ray_info),
                    .ray_idx = idx + pre_init_ray_size,
                    .sample_idx = inp_state.sample_idx,
                    .x = inp_state.x,
                    .y = inp_state.y,
                };
              } else {
                return inp_state;
              }
            }();

            unsigned end_ray =
                sample_state.ray_idx + sample_state.render_state.num_samples();

            auto ray_span = rays_in_span.slice(sample_state.ray_idx, end_ray);
            auto last_ray = sample_state.render_state.get_last_ray(ray_span);
            auto intersections =
                intersections_span.slice(sample_state.ray_idx, end_ray);

            unsigned location =
                kernel::get_location(sample_state.x, sample_state.y, x_dim);
            auto rng_state =
                rng.state_from_saved(sample_state.sample_idx, location,
                                     rng_states_in_span[overall_state_idx]);
            auto next = rendering_equation_iteration(
                sample_state.render_state, rng_state, last_ray, intersections,
                render_settings, components);

            next.visit_tagged([&](auto tag, auto &value) {
              constexpr auto type = tag;
              if constexpr (type ==
                            integrate::IterationOutputType::NextIteration) {
                // standard atomic idx fetch add approach (like in filter)
                const unsigned new_state_idx = state_idx_span[0].fetch_add(1);
                const unsigned new_ray_idx =
                    ray_idx_span[0].fetch_add(value.rays.size());
                states_out_span[new_state_idx] = {
                    .render_state = value.state,
                    .ray_idx = new_ray_idx,
                    .sample_idx = sample_state.sample_idx,
                    .x = sample_state.x,
                    .y = sample_state.y,
                };
                rng_states_out_span[new_state_idx] = rng_state.save();
                debug_assert_assume(value.rays.size() <= max_rays_per_sample);
#pragma unroll max_rays_per_sample
                for (unsigned i = 0; i < value.rays.size(); ++i) {
                  rays_out_span[i + new_ray_idx] = value.rays[i];
                }
              } else {
                auto &float_rgb = float_rgb_atomic[location];
#pragma unroll FloatRGB{}.size
                for (unsigned i = 0; i < value.size; ++i) {
                  float_rgb[i].fetch_add(value[i]);
                }
                static_assert(type == integrate::IterationOutputType::Finished);
              }
            });
          });
    };

    always_assert(rng_states_in.size() ==
                  (states_in.size() + state.initial_sample_info.size()));

    unsigned max_out_states = rng_states_in.size();
    rays_out.resize(max_out_states * max_rays_per_sample);
    states_out.resize(max_out_states);
    rng_states_out.resize(max_out_states);

    shade(tag_v<false>, Span{states_in}.as_const());
    shade(tag_v<true>, Span{state.initial_sample_info}.as_const());

    rays_out.resize(state.ray_idx.get().as_inner());

    num_states = state.state_idx.get().as_inner();
    always_assert(num_states <= max_num_samples_per_launch);

    states_out.resize(num_states);
    rng_states_out.resize(num_states);

    if (inp.show_progress) {
      progress_bar += end_block_init - start_block_init;
      progress_bar.display();
    }

    start_block_init = end_block_init;
    parity = !parity;
  }

  auto start = thrust::make_counting_iterator(0u);
  auto end = start + inp.x_dim * inp.y_dim;

  unsigned x_dim = inp.x_dim;

  if (!inp.output_as_bgra_32) {
    float_rgb_out.resize(inp.x_dim * inp.y_dim);
  }

  BaseItems base = {
      .output_as_bgra_32 = inp.output_as_bgra_32,
      .samples_per = inp.samples_per,
      .bgra_32 = inp.bgra_32,
      .float_rgb = float_rgb_out,
  };

  thrust::for_each(ThrustData<exec>().execution_policy(), start, end,
                   [=] HOST_DEVICE(unsigned idx) {
                     FloatRGB value;
#pragma unroll FloatRGB{}.size
                     for (unsigned i = 0; i < value.size; ++i) {
                       value[i] = float_rgb_atomic[idx][i].as_inner();
                     }

                     unsigned x = idx % x_dim;
                     unsigned y = idx / x_dim;

                     assign_output(base, x_dim, 0, 1, x, y, value);
                   });

  if (inp.show_progress) {
    progress_bar.done();
  }
}
} // namespace streaming
} // namespace integrate_image
} // namespace detail
} // namespace render
