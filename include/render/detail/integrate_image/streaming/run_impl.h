#pragma once

#include "execution_model/thrust_data.h"
#include "integrate/rendering_equation_iteration.h"
#include "kernel/atomic.h"
#include "kernel/kernel_launch.h"
#include "kernel/location_info.h"
#include "kernel/make_no_interactor_launchable.h"
#include "kernel/progress_bar_launch.h"
#include "lib/info/timer.h"
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
template <ExecutionModel exec, typename Inp, typename State>
void initialize(const Inp &inp, State &state,
                const kernel::WorkDivision &division, unsigned start_block_init,
                unsigned end_block_init, unsigned max_num_samples_to_init,
                bool parity) {
  auto &rays_in = state.rays_in_and_out[parity];
  auto &rng_states_in = state.rng_states_in_and_out[parity];

  state.local_idx.set(0);

  unsigned existing_rays = rays_in.size();
  unsigned existing_states = rng_states_in.size();

  unsigned max_new_rays = existing_rays + max_num_samples_to_init;
  unsigned max_new_states = existing_states + max_num_samples_to_init;

  rays_in.resize(max_new_rays);
  rng_states_in.resize(max_new_states);
  state.initial_sample_info.resize(max_num_samples_to_init);

  Span rays_in_span = Span{rays_in}.slice_from(existing_rays).as_unsized();
  Span rng_states_in_span =
      Span{rng_states_in}.slice_from(existing_states).as_unsized();
  Span initial_sample_info = Span{state.initial_sample_info}.as_unsized();
  Span local_idx_span = state.local_idx.span().as_unsized();

  auto rng = inp.items.rng;

  // TODO: dedup with megakernel
  auto sample_value = inp.sample_spec.visit_tagged(
      [&](auto tag, const auto &spec) -> SampleValue {
        if constexpr (tag == SampleSpecType::SquareImage) {
          return {tag, spec.film_to_world};
        } else {
          static_assert(tag == SampleSpecType::InitialRays);
          return {tag, spec};
        }
      });

  // TODO: fix thrust data!!!
  kernel::KernelLaunch<exec>::run(
      ThrustData<exec>{}, division, start_block_init, end_block_init,
      kernel::make_no_interactor_launchable(
          [=] HOST_DEVICE(const kernel::WorkDivision &division,
                          const kernel::GridLocationInfo &info, unsigned,
                          unsigned, const auto &, const auto &) {
            unsigned location =
                kernel::get_location(info.x, info.y, division.x_dim());
            // this atomic usage is pretty gross, but its simple
            // and faster enough I guess (this kernel isn't limiting...)
            unsigned base_local_idx = local_idx_span[0].fetch_add(
                info.end_sample - info.start_sample);
            for (unsigned sample_idx = info.start_sample;
                 sample_idx < info.end_sample; ++sample_idx) {
              auto rng_state = rng.get_generator(sample_idx, location);
              auto [ray, ray_info] = initial_ray_sample(
                  rng_state, info.x, info.y, division.x_dim(), division.y_dim(),
                  sample_value);
              unsigned local_idx =
                  base_local_idx + (sample_idx - info.start_sample);
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

  unsigned num_samples = state.local_idx.get().as_inner();

  unsigned new_rays = existing_rays + num_samples;
  unsigned new_states = existing_states + num_samples;

  rays_in.resize(new_rays);
  rng_states_in.resize(new_states);
  state.initial_sample_info.resize(num_samples);
}

template <ExecutionModel exec, typename Inp>
void copy_to_output(
    const Inp &inp, unsigned x_dim, unsigned num_locs,
    ExecVector<exec, BGRA32> &bgra_32,
    ExecVector<exec, FloatRGB> &float_rgb_out,
    Span<std::array<kernel::Atomic<exec, float>, 3>> float_rgb_atomic) {
  auto start = thrust::make_counting_iterator(0u);
  auto end = start + num_locs;

  if (inp.output_type == OutputType::OutputPerStep) {
    unreachable(); // NYI
  } else if (inp.output_type == OutputType::BGRA) {
    bgra_32.resize(num_locs);
    float_rgb_out.clear();
  } else {
    always_assert(inp.output_type == OutputType::FloatRGB);
    bgra_32.clear();
    float_rgb_out.resize(num_locs);
  }

  BaseItems base = {
      .output_as_bgra_32 = inp.output_type == OutputType::BGRA,
      .samples_per = inp.samples_per,
      .bgra_32 = bgra_32,
      .float_rgb = float_rgb_out,
  };

  thrust::for_each(ThrustData<exec>().execution_policy(), start, end,
                   [=] HOST_DEVICE(unsigned idx) {
                     FloatRGB value;
#pragma unroll FloatRGB::size
                     for (unsigned i = 0; i < FloatRGB::size; ++i) {
                       value[i] = float_rgb_atomic[idx][i].as_inner();
                     }

                     unsigned x = idx % x_dim;
                     unsigned y = idx / x_dim;

                     assign_output_single(base, x_dim, x, y, value);
                   });
}
} // namespace detail

template <ExecutionModel exec>
template <ExactSpecializationOf<Items> Items,
          intersectable_scene::BulkIntersector I>
requires std::same_as<typename Items::InfoType, typename I::InfoType> Output
Run<exec>::run(
    Inputs<Items> inp, I &intersector,
    State<exec, Items::C::max_num_light_samples(), typename Items::R> &state,
    const StreamingSettings &settings, ExecVector<exec, BGRA32> &bgra_32,
    ExecVector<exec, FloatRGB> &float_rgb_out,
    HostVector<ExecVector<exec, FloatRGB>> &output_per_step_rgb_out) {
  using namespace detail;

  // TODO: dedup with mega_kernel?
  kernel::WorkDivision init_division = inp.sample_spec.visit_tagged(
      [&](auto tag, const auto &spec) -> kernel::WorkDivision {
        if constexpr (tag == SampleSpecType::SquareImage) {
          return {
              settings.computation_settings.init_samples_division,
              inp.samples_per,
              spec.x_dim,
              spec.y_dim,
          };
        } else {
          return {
              // TODO: is this a sane work division in this case?
              // TODO: should we really be using WorkDivision when it
              // isn't a grid (it is convenient...) - see also reduce
              settings.computation_settings.init_samples_division,
              inp.samples_per,
              unsigned(spec.size()),
              1,
          };
        }
      });

  if (inp.output_type == OutputType::OutputPerStep) {
    unreachable(); // NYI!!!
  }

  constexpr unsigned max_num_light_samples_per =
      Items::C::max_num_light_samples();

  // add 1 for next bounce
  constexpr unsigned max_rays_per_sample = max_num_light_samples_per + 1;

  unsigned max_num_extra_samples_per_thread =
      init_division.n_threads_per_unit_extra() > 0 ? 1 : 0;
  unsigned max_samples_per_thread = init_division.base_samples_per_thread() +
                                    max_num_extra_samples_per_thread;

  bool has_multiple_sample_blocks = init_division.num_sample_blocks() > 1;

  // In the case where we have samples over multiple different blocks, we
  // can upperbound with max_samples_per_thread.
  // Otherwise, we can just use samples per.
  unsigned max_samples_per_loc =
      has_multiple_sample_blocks
          ? init_division.sample_block_size() * max_samples_per_thread
          : inp.samples_per;

  unsigned num_locs_per_block =
      init_division.x_block_size() * init_division.y_block_size();

  unsigned max_samples_per_init_block =
      num_locs_per_block * max_samples_per_loc;

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

  unsigned overall_num_locations =
      init_division.x_dim() * init_division.y_dim();

  state.float_rgb.resize(overall_num_locations);
  thrust::fill(state.float_rgb.begin(), state.float_rgb.end(),
               std::array<kernel::Atomic<exec, float>, 3>{});
  Span float_rgb_atomic = Span{state.float_rgb}.as_unsized();

  unsigned num_states = 0;

  Timer total_init(std::nullopt);
  Timer total_intersect(std::nullopt);
  Timer total_shade(std::nullopt);

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
      total_init.start();
      initialize<exec>(inp, state, init_division, start_block_init,
                       attempt_end_block_init, max_num_samples_to_init, parity);
      total_init.stop();
      always_assert(state.initial_sample_info.size() > 0);
      end_block_init = attempt_end_block_init;
    }

    total_intersect.start();
    auto intersections_span = intersector.get_intersections(Span{rays_in});
    total_intersect.stop();

    states_out.resize(rng_states_in.size());

    using SampleStateT = SampleState<max_num_light_samples_per>;

    auto shade = [&]<typename T>(auto tag,
                                 SpanSized<const T> sample_states_span_in) {
      constexpr bool is_init = tag;

      auto start = thrust::make_counting_iterator(0u);
      auto end = start + sample_states_span_in.size();

      auto sample_states_span = sample_states_span_in.as_unsized();

      Span states_out_span = Span{states_out}.as_unsized();
      // Span rng_states_out_span = Span{rng_states_out}.as_unsized();
      Span rng_states_out_span = Span{rng_states_out};
      Span rays_out_span = Span{rays_out}.as_unsized();
      Span rng_states_in_span = Span{rng_states_in}.as_unsized().as_const();
      Span rays_in_span = Span{rays_in}.as_unsized().as_const();
      Span state_idx_span = state.state_idx.span().as_unsized();
      Span ray_idx_span = state.ray_idx.span().as_unsized();
      auto components = inp.items.components;
      auto rng = inp.items.rng;

      unsigned x_dim = init_division.x_dim();
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
                static_assert(type == integrate::IterationOutputType::Finished);

                auto &float_rgb = float_rgb_atomic[location];
#pragma unroll FloatRGB::size
                for (unsigned i = 0; i < FloatRGB::size; ++i) {
                  float_rgb[i].fetch_add(value[i]);
                }
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

    total_shade.start();
    shade(tag_v<false>, Span{states_in}.as_const());
    shade(tag_v<true>, Span{state.initial_sample_info}.as_const());
    total_shade.stop();

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

  copy_to_output<exec>(inp, init_division.x_dim(), overall_num_locations,
                       bgra_32, float_rgb_out, float_rgb_atomic);

  if (inp.show_progress) {
    progress_bar.done();
  }

  if (inp.show_times) {
    total_init.report("total init");
    total_intersect.report("total intersect");
    total_shade.report("total shade");
  }

  if (inp.output_type == OutputType::BGRA) {
    return {tag_v<OutputType::BGRA>, bgra_32};
  } else {
    // OutputPerStep NYI!!!
    always_assert(inp.output_type == OutputType::FloatRGB);

    return {tag_v<OutputType::FloatRGB>, float_rgb_out};
  }
}
} // namespace streaming
} // namespace integrate_image
} // namespace detail
} // namespace render
