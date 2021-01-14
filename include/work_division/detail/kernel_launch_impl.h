#pragma once

#include "boost/hana/unpack.hpp"
#include "lib/cuda/utils.h"
#include "meta/tuple.h"
#include "work_division/launchable.h"
#include "work_division/thread_interactor.h"
#include "work_division/work_division.h"
#include "work_division/work_division_impl.h"

namespace work_division {
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
} // namespace work_division
