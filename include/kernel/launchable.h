#pragma once

#include "kernel/work_division.h"

#include <concepts>

namespace kernel {
template <typename T>
concept LaunchableBlockRef = requires(T &t, const WorkDivision &division,
                                      const GridLocationInfo &info,
                                      const unsigned block_idx,
                                      const unsigned thread_idx) {
  t(division, info, block_idx, thread_idx);
};

// Launchable function is allowed to have internal state which it mutates,
// but it must be copyable. A different copy will be used for each thread.
template <typename T>
concept Launchable = requires(T &t, const WorkDivision &division,
                              const unsigned block_idx) {
  requires std::is_copy_constructible_v<T>;
  typename T::BlockRef;
  requires LaunchableBlockRef<typename T::BlockRef>;
  { t.block_init(division, block_idx) } -> std::same_as<typename T::BlockRef>;
};
} // namespace kernel
