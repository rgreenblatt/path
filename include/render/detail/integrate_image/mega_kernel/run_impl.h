#pragma once

#include "kernel/kernel_launch.h"
#include "kernel/make_runtime_constants_reduce_launchable.h"
#include "kernel/progress_bar_launch.h"
#include "lib/assert.h"
#include "lib/integer_division_utils.h"
#include "meta/all_values/dispatch.h"
#include "meta/all_values/impl/integral.h"
#include "meta/all_values/impl/tuple.h"
#include "render/detail/integrate_image/items.h"
#include "render/detail/integrate_image/mega_kernel/integrate_pixel.h"
#include "render/detail/integrate_image/mega_kernel/reduce_assign_output.h"
#include "render/detail/integrate_image/mega_kernel/reduce_float_rgb.h"
#include "render/detail/integrate_image/mega_kernel/run.h"
#include "render/mega_kernel_settings.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
namespace detail {
template <ExecutionModel exec>
unsigned
max_blocks_per_launch(const MegaKernelSettings::ComputationSettings &settings) {
  if constexpr (exec == ExecutionModel::GPU) {
    return settings.max_blocks_per_launch_gpu;
  } else {
    if (debug_build) {
      return 1;
    } else {
      return settings.max_blocks_per_launch_cpu;
    }
  }
}
} // namespace detail

template <ExecutionModel exec>
template <ExactSpecializationOf<Items> Items, intersect::Intersectable I>
requires std::same_as<typename Items::InfoType, typename I::InfoType> Output
Run<exec>::run(
    ThrustData<exec> &data, Inputs<Items> inp, const I &intersectable,
    const MegaKernelSettings &settings, ExecVector<exec, BGRA32> &bgra_32,
    std::array<ExecVector<exec, FloatRGB>, 2> &float_rgb,
    std::array<HostVector<ExecVector<exec, FloatRGB>>, 2> &output_per_step_rgb,
    HostVector<Span<FloatRGB>> &output_per_step_rgb_spans_out) {
  kernel::WorkDivision division = inp.sample_spec.visit_tagged(
      [&](auto tag, const auto &spec) -> kernel::WorkDivision {
        if constexpr (tag == SampleSpecType::SquareImage) {
          return {
              settings.computation_settings.render_work_division,
              inp.samples_per,
              spec.x_dim,
              spec.y_dim,
          };
        } else {
          static_assert(tag == SampleSpecType::InitialRays ||
                        tag == SampleSpecType::InitialIdxAndDir);
          return {
              // TODO: is this a sane work division in this case?
              // TODO: should we really be using WorkDivision when it
              // isn't a grid (it is convenient...) - see also reduce
              settings.computation_settings.render_work_division,
              inp.samples_per,
              unsigned(spec.size()),
              1,
          };
        }
      });

  unsigned num_locs = division.x_dim() * division.y_dim();

  if (inp.output_type == OutputType::BGRA) {
    bgra_32.resize(num_locs);
  }
  if (inp.output_type == OutputType::OutputPerStep) {
    for (auto &vec : output_per_step_rgb[0]) {
      vec.resize(division.num_sample_blocks() * num_locs);
    }
  } else if (division.num_sample_blocks() != 1 ||
             inp.output_type == OutputType::FloatRGB) {
    float_rgb[0].resize(division.num_sample_blocks() * num_locs);
  }

  BaseItems base{
      .output_as_bgra_32 = inp.output_type == OutputType::BGRA,
      .samples_per = inp.samples_per,
      .bgra_32 = bgra_32,
      .float_rgb = float_rgb[0],
  };

  bool is_output_per_step = inp.output_type == OutputType::OutputPerStep;

  dispatch(
      MetaTuple<bool, bool>{inp.sample_spec.type() ==
                                SampleSpecType::InitialIdxAndDir,
                            is_output_per_step},
      [&](auto values_in) {
        constexpr MetaTuple<bool, bool> values = values_in;
        constexpr bool is_initial_idx = values[boost::hana::int_c<0>];
        constexpr bool is_output_per_step = values[boost::hana::int_c<1>];

        auto get_sampler = [&]() {
          if constexpr (is_initial_idx) {
            return [initial = inp.sample_spec.get(
                        tag_v<SampleSpecType::InitialIdxAndDir>),
                    idx_to_info =
                        inp.idx_to_info](const kernel::WorkDivision &,
                                         const kernel::GridLocationInfo &info) {
              return [&](auto &) {
                return initial_intersection_sample(initial[info.x],
                                                   idx_to_info);
              };
            };
          } else {
            auto sample_value = inp.sample_spec.visit_tagged(
                [&](auto tag, const auto &spec) -> SampleValue {
                  if constexpr (tag == SampleSpecType::SquareImage) {
                    return {tag, spec.film_to_world};
                  } else if constexpr (tag == SampleSpecType::InitialRays) {
                    return {tag, spec};
                  } else {
                    static_assert(tag == SampleSpecType::InitialIdxAndDir);
                    // Covered by other case!
                    unreachable();
                  }
                });

            return [sample_value](const kernel::WorkDivision &division,
                                  const kernel::GridLocationInfo &info) {
              return [&](auto &rng) {
                return initial_ray_sample(rng, info.x, info.y, division.x_dim(),
                                          division.y_dim(), sample_value);
              };
            };
          }
        }();

        kernel::progress_bar_launch(
            division,
            detail::max_blocks_per_launch<exec>(settings.computation_settings),
            inp.show_progress, [&](unsigned start, unsigned end) {
              auto items = inp.items;

              // TODO: remove exec vecs!
              unsigned samples_per = base.samples_per;
              ExecVector<exec, Span<FloatRGB>> span_holder(
                  output_per_step_rgb[0].size());
              for (unsigned i = 0; i < output_per_step_rgb[0].size(); ++i) {
                span_holder[i] = output_per_step_rgb[0][i];
              }
              SpanSized<const Span<FloatRGB>> output_per_step_rgb_spans =
                  span_holder;
              ExecVector<exec, FloatRGB> buf_values(
                  division.block_size() * (end - start) *
                  output_per_step_rgb_spans.size());
              Span<FloatRGB> buf_slice = buf_values;

              kernel::KernelLaunch<exec>::run(
                  data, division, start, end,
                  kernel::make_runtime_constants_reduce_launchable<exec,
                                                                   FloatRGB>(
                      is_output_per_step ? output_per_step_rgb_spans.size() : 1,
                      [=](const kernel::WorkDivision &division,
                          const kernel::GridLocationInfo &info,
                          const unsigned block_idx, unsigned thread_idx,
                          const auto &, auto &reducer) {
                        if constexpr (is_output_per_step) {
                          debug_assert(block_idx >= start);
                          debug_assert(block_idx < end);
                          unsigned start_buf =
                              output_per_step_rgb_spans.size() *
                              (division.block_size() * (block_idx - start) +
                               thread_idx);

                          auto buf = buf_slice.slice(
                              start_buf,
                              start_buf + output_per_step_rgb_spans.size());
                          std::fill(buf.begin(), buf.end(), FloatRGB::Zero());

                          integrate_pixel<is_output_per_step>(
                              items, intersectable, division, info,
                              get_sampler(division, info), buf);

                          for (unsigned i = 0; i < buf.size(); ++i) {
                            reduce_assign_output(
                                reducer[i],
                                {.output_as_bgra_32 = false,
                                 .samples_per = samples_per,
                                 .bgra_32 = {},
                                 .float_rgb = output_per_step_rgb_spans[i]},
                                division, block_idx, info.x, info.y, buf[i]);
                          }
                        } else {
                          const FloatRGB float_rgb =
                              integrate_pixel<is_output_per_step>(
                                  items, intersectable, division, info,
                                  get_sampler(division, info), {});

                          reduce_assign_output(reducer[0], base, division,
                                               block_idx, info.x, info.y,
                                               float_rgb);
                        }
                      }));
            });
      });

  auto run_reduce = [&](ExecVector<exec, FloatRGB> &l,
                        ExecVector<exec, FloatRGB> &r) {
    auto float_rgb_reduce_out = ReduceFloatRGB<exec>::run(
        data, settings.computation_settings.reduce_work_division,
        base.output_as_bgra_32, division.num_sample_blocks(), inp.samples_per,
        &l, &r, bgra_32);
    always_assert(float_rgb_reduce_out != nullptr);
    always_assert(float_rgb_reduce_out == &l || float_rgb_reduce_out == &r);

    return float_rgb_reduce_out;
  };

  if (is_output_per_step) {
    always_assert(!base.output_as_bgra_32);
    // This might not be peak efficiency!
    output_per_step_rgb_spans_out.resize(output_per_step_rgb[0].size());
    for (unsigned i = 0; i < output_per_step_rgb[0].size(); ++i) {
      output_per_step_rgb_spans_out[i] =
          *run_reduce(output_per_step_rgb[0][i], output_per_step_rgb[1][i]);
    }

    return {tag_v<OutputType::OutputPerStep>, output_per_step_rgb_spans_out};
  } else {
    auto float_rgb_reduce_out = run_reduce(float_rgb[0], float_rgb[1]);

    if (inp.output_type == OutputType::BGRA) {
      return {tag_v<OutputType::BGRA>, bgra_32};
    } else {
      // OutputPerStep NYI!!!
      always_assert(inp.output_type == OutputType::FloatRGB);

      return {tag_v<OutputType::FloatRGB>, *float_rgb_reduce_out};
    }
  }
}
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
