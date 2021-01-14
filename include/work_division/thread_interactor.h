#pragma once

#include "work_division/work_division.h"

namespace work_division {
template <typename T>
concept ThreadInteractor = requires(const WorkDivision &division, T &t_mut,
                                    unsigned thread_idx) {
  std::movable<T>;
  T{division};
  t_mut.set_thread_idx(thread_idx);
};
} // namespace work_division
