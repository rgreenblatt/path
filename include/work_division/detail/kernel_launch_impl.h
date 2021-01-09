#pragma once

#include "lib/cuda/utils.h"
#include "work_division/work_division.h"
#include "work_division/work_division_impl.h"

namespace work_division {
namespace detail {
template <typename F>
HOST_DEVICE void kernel_launch_run(const WorkDivision &division,
                                   const unsigned block_idx,
                                   const unsigned thread_idx, const F &f) {
  auto [info, exit] = division.get_thread_info(block_idx, thread_idx);

  if (exit) {
    return;
  }

  f(division, info, block_idx, thread_idx);
}
} // namespace detail
} // namespace work_division
