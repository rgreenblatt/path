#pragma once

#include "kernel/thread_interactor.h"
#include "kernel/work_division.h"
#include "meta/container_concepts.h"

namespace kernel {
// Launchable function is allowed to have internal state which it mutates,
// but it must be copyable. A different copy will be used for each thread.
template <typename F, typename... Interactors>
concept Launchable = requires(F &f, const WorkDivision &division,
                              const GridLocationInfo &info,
                              const unsigned block_idx,
                              const unsigned thread_idx) {
  requires(... && ThreadInteractor<Interactors>);
  requires std::is_copy_constructible_v<F>;
  f(division, info, block_idx, thread_idx, std::declval<Interactors &>()...);
};
} // namespace kernel
