#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "integrate/rendering_equation_state.h"
#include "intersectable_scene/intersectable_scene.h"
#include "kernel/kernel_launch.h"
#include "kernel/location_info.h"
#include "kernel/work_division.h"
#include "lib/integer_division_utils.h"
#include "render/detail/initial_ray_sample.h"
#include "render/general_settings.h"
#include "rng/rng.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <ExecutionModel exec, unsigned max_num_light_samples, rng::RngRef R>
class BulkRunner {
public:
  BulkRunner() {}

  using State = integrate::RenderingEquationState<max_num_light_samples>;

  template <intersectable_scene::BulkIntersector I>
  void run_bulk(const kernel::WorkDivision &division,
                const GeneralSettings &settings, I &intersector, const R &rng,
                const Eigen::Affine3f &film_to_world, bool show_progress) {
    always_assert(division.block_size() <= intersector.max_size());

    unsigned total_grid = division.total_num_blocks();

    unsigned max_launch_size =
        std::min(intersector.max_size() / division.block_size(),
                 settings.computation_settings.max_blocks_per_launch);

    unsigned num_launches = ceil_divide(total_grid, max_launch_size);
    unsigned blocks_per = total_grid / num_launches;

    always_assert(static_cast<uint64_t>(blocks_per) * division.block_size() <
                  static_cast<uint64_t>(std::numeric_limits<unsigned>::max()));

    ProgressBar progress_bar(num_launches, 70);
    if (show_progress) {
      progress_bar.display();
    }

    for (unsigned i = 0; i < num_launches; i++) {
      unsigned start = i * blocks_per;
      unsigned end = std::min((i + 1) * blocks_per, total_grid);
      unsigned grid = end - start;

      unsigned initial_size = grid * division.block_size();
      auto ray_writer = intersector.ray_writer(initial_size);
      state_.resize(initial_size);
      rng_state_.resize(initial_size);

      Span<State> state = state_;
      Span<typename R::State> rng_state = rng_state_;

      // be precise with copies...
      auto get_local_idx = [=](const kernel::WorkDivision &division,
                               unsigned block_idx, unsigned thread_idx) {
        return (block_idx - start) * division.block_size() + thread_idx;
      };

      // initialize
      // TODO: SPEED: change division used for this kernel???
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
              rng_state[local_idx] =
                  rng.get_generator(sample_idx, loc_info.location);
              auto initial_sample = initial_ray_sample(
                  rng_state[local_idx], info.x, info.y, division.x_dim(),
                  division.y_dim(), film_to_world);
              ray_writer(local_idx, initial_sample.ray);
              state[local_idx] = State::initial_state(initial_sample);
            }
          });

      bool has_remaining_samples = true;

      const auto &components = inp.components;

      while (has_remaining_samples) {
        auto intersections = intersector.get_intersections();
        kernel::KernelLaunch<exec>::run(
            division, start, end,
            [=] HOST_DEVICE(const kernel::WorkDivision &division,
                            const kernel::GridLocationInfo &,
                            unsigned block_idx, unsigned thread_idx) {
              const unsigned local_idx =
                  get_local_idx(division, block_idx, thread_idx);
              rendering_equation_iteration(
                  state[local_idx], rng_state[local_idx], TODO_intersections,
                  rendering_settings, components);
            });
      }

      if (show_progress) {
        ++progress_bar;
        progress_bar.display();
      }
    }

    if (show_progress) {
      progress_bar.done();
    }

    unreachable();
  }

private:
  ExecVector<exec, State> state_;
  ExecVector<exec, typename R::State> rng_state_;
};
} // namespace detail
} // namespace render
