#pragma once

#include "execution_model/execution_model.h"
#include "kernel/launchable.h"
#include "kernel/runtime_constants_reducer.h"
#include "kernel/thread_interactor_launchable.h"

namespace kernel {
template <ExecutionModel exec, typename Reduced, typename F>
constexpr auto make_runtime_constants_reduce_launchable(unsigned count,
                                                        const F &callable) {
  return ThreadInteractorLaunchableNoExtraInp<
      kernel::RuntimeConstantsReducer<exec, Reduced>, F>{
      .inp = {}, .interactor = {count}, .callable = callable};
}
} // namespace kernel
