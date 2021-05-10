#pragma once

#include "kernel/kernel_launch.h"
#include "kernel/make_runtime_constants_reduce_launchable.h"
#include "kernel/progress_bar_launch.h"
#include "lib/assert.h"
#include "lib/integer_division_utils.h"
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
requires std::same_as<typename Items::InfoType, typename I::InfoType>
    Output Run<exec>::run(Inputs<Items> inp, const I &intersectable,
                          const MegaKernelSettings &settings,
                          ExecVector<exec, BGRA32> &bgra_32,
                          std::array<ExecVector<exec, FloatRGB>, 2> &float_rgb,
                          std::array<HostVector<ExecVector<exec, FloatRGB>>, 2>
                              &output_per_step_rgb) {
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

  if (inp.output_type == OutputType::OutputPerStep) {
    unreachable(); // NYI!!!
  }

  BaseItems base{
      .output_as_bgra_32 = inp.output_type == OutputType::BGRA,
      .samples_per = inp.samples_per,
      .bgra_32 = bgra_32,
      .float_rgb = float_rgb[0],
  };

  auto sample_value = inp.sample_spec.visit_tagged(
      [&](auto tag, const auto &spec) -> SampleValue {
        if constexpr (tag == SampleSpecType::SquareImage) {
          return {tag, spec.film_to_world};
        } else {
          static_assert(tag == SampleSpecType::InitialRays);
          return {tag, spec};
        }
      });

  kernel::progress_bar_launch(
      division,
      detail::max_blocks_per_launch<exec>(settings.computation_settings),
      inp.show_progress, [&](unsigned start, unsigned end) {
        auto items = inp.items;

        kernel::KernelLaunch<exec>::run(
            division, start, end,
            kernel::make_runtime_constants_reduce_launchable<exec, FloatRGB>(
                [=] HOST_DEVICE(const kernel::WorkDivision &division,
                                const kernel::GridLocationInfo &info,
                                const unsigned block_idx, unsigned,
                                const auto &, auto &reducer) {
                  auto float_rgb = integrate_pixel(
                      items, sample_value, intersectable, division, info);

                  reduce_assign_output(reducer, base, division, block_idx,
                                       info.x, info.y, float_rgb);
                }));
      });

  auto float_rgb_reduce_out = ReduceFloatRGB<exec>::run(
      settings.computation_settings.reduce_work_division,
      base.output_as_bgra_32, division.num_sample_blocks(), inp.samples_per,
      &float_rgb[0], &float_rgb[1], bgra_32);
  always_assert(float_rgb_reduce_out != nullptr);
  always_assert(float_rgb_reduce_out == &float_rgb[0] ||
                float_rgb_reduce_out == &float_rgb[1]);

  if (inp.output_type == OutputType::BGRA) {
    return {tag_v<OutputType::BGRA>, bgra_32};
  } else {
    // OutputPerStep NYI!!!
    always_assert(inp.output_type == OutputType::FloatRGB);

    return {tag_v<OutputType::FloatRGB>, *float_rgb_reduce_out};
  }
}
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
