#pragma once

#include "kernel/work_division.h"

namespace kernel {
template <typename T>
concept ThreadInteractor = requires(const WorkDivision &division, T &t_mut,
                                    unsigned thread_idx) {
  std::movable<T>;
  T{division};
  t_mut.set_thread_idx(thread_idx);
};
} // namespace kernel
