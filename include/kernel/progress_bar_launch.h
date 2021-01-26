#pragma once

#include "execution_model/execution_model.h"
#include "kernel/launchable.h"
#include "kernel/work_division.h"
#include "lib/assert.h"
#include "lib/integer_division_utils.h"

#include <cli/ProgressBar.hpp>

namespace kernel {
// TODO: better name?
template <typename F>
void progress_bar_launch(const WorkDivision &division, unsigned max_launch_size,
                         bool show_progress, F &&launches) {

  unsigned total_grid = division.total_num_blocks();

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

    launches(start, end);

    if (show_progress) {
      ++progress_bar;
      progress_bar.display();
    }
  }

  if (show_progress) {
    progress_bar.done();
  }
}
} // namespace kernel
