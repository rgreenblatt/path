#pragma once

#include "kernel/launchable.h"
#include "kernel/thread_interactor_launchable.h"
#include "kernel/tuple_thread_interactor.h"

namespace kernel {
template <typename F>
constexpr auto make_no_interactor_launchable(const F &callable) {
  return ThreadInteractorLaunchableNoExtraInp<
      kernel::TupleThreadInteractor<EmptyExtraInp>, F>{{}, {}, callable};
}
} // namespace kernel
