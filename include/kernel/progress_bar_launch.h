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
void progress_bar_launch(const WorkDivision &division,
                         unsigned max_blocks_per_launch, bool show_progress,
                         F &&launches) {
  unsigned total_num_blocks = division.total_num_blocks();

  unsigned num_launches = ceil_divide(total_num_blocks, max_blocks_per_launch);
  unsigned blocks_per = total_num_blocks / num_launches;

  always_assert(static_cast<uint64_t>(blocks_per) * division.block_size() <
                static_cast<uint64_t>(std::numeric_limits<unsigned>::max()));

  ProgressBar progress_bar(total_num_blocks, 70);
  if (show_progress) {
    progress_bar.display();
  }

  for (unsigned i = 0; i < num_launches; i++) {
    unsigned start = i * blocks_per;
    unsigned end = std::min((i + 1) * blocks_per, total_num_blocks);

    launches(start, end);

    if (show_progress) {
      progress_bar += end - start;
      progress_bar.display();
    }
  }

  if (show_progress) {
    progress_bar.done();
  }
}
} // namespace kernel
