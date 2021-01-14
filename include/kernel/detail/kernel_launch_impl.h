#pragma once

#include "boost/hana/unpack.hpp"
#include "kernel/launchable.h"
#include "kernel/thread_interactor.h"
#include "kernel/work_division.h"
#include "kernel/work_division_impl.h"
#include "lib/cuda/utils.h"
#include "meta/tuple.h"

namespace kernel {
namespace detail {
template <ThreadInteractor... Interactors, Launchable<Interactors...> F>
HOST_DEVICE void kernel_launch_run(const WorkDivision &division,
                                   const unsigned block_idx,
                                   const unsigned thread_idx, F &f,
                                   MetaTuple<Interactors...> &interactors_tup) {
  auto info = division.get_thread_info(block_idx, thread_idx);

  if (info.exit) {
    return;
  }

  boost::hana::unpack(interactors_tup, [&](auto &...interactors) {
    (interactors.set_thread_idx(thread_idx), ...);
    f(division, info.info, block_idx, thread_idx, interactors...);
  });
}
} // namespace detail
} // namespace kernel
