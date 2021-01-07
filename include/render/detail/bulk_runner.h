#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "integrate/rendering_equation_state.h"
#include "intersectable_scene/intersectable_scene.h"
#include "lib/integer_division_utils.h"
#include "render/general_settings.h"
#include "rng/rng.h"
#include "work_division/work_division.h"

#include <cli/ProgressBar.hpp>

namespace render {
template <ExecutionModel exec, std::semiregular State, rng::RngState R>
class BulkRunner {
public:
  BulkRunner() {}

  template <intersectable_scene::IntersectableScene S>
  requires(!S::individually_intersectable) void run_bulk(
      const work_division::WorkDivision &division,
      const GeneralSettings &settings, S &scene,
      const Eigen::Affine3f &film_to_world, bool show_progress) {
    always_assert(division.block_size() <= scene.max_size());

    unsigned total_grid = division.total_num_blocks();

    unsigned max_launch_size =
        std::min(scene.max_size() / division.block_size(),
                 settings.computation_settings.max_blocks_per_launch);

    unsigned num_launches = ceil_divide(total_grid, max_launch_size);
    unsigned blocks_per = total_grid / num_launches;

    ProgressBar progress_bar(num_launches, 70);
    if (show_progress) {
      progress_bar.display();
    }

    for (unsigned i = 0; i < num_launches; i++) {
      unsigned start = i * blocks_per;
      unsigned end = std::min((i + 1) * blocks_per, total_grid);
      unsigned grid = end - start;

      unsigned initial_size = grid * division.block_size();
      auto ray_writer = scene.ray_writer(initial_size);
      state_.resize(initial_size);
      rng_state_.resize(initial_size);

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
  ExecVector<exec, R> rng_state_;
};
} // namespace render
