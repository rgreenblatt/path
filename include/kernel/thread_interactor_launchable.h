#pragma once

#include "kernel/grid_location_info.h"
#include "kernel/launchable.h"
#include "kernel/thread_interactor.h"
#include "kernel/work_division.h"
#include "meta/mock.h"

namespace kernel {
struct EmptyExtraInp {};

template <typename T, typename ExtraInp, typename ThreadRef>
concept ThreadCallableForInteractor = requires(
    T &callable, const WorkDivision &division, const GridLocationInfo &info,
    const unsigned block_idx, const unsigned thread_idx, const ExtraInp &inp,
    ThreadRef &interactor) {
  requires std::copyable<T>;
  callable(division, info, block_idx, thread_idx, inp, interactor);
};

struct MockThreadCallable : MockCopyable {
  template <typename ExtraInp, typename ThreadRef>
  void operator()(const WorkDivision &division, const GridLocationInfo &info,
                  const unsigned block_idx, const unsigned thread_idx,
                  const ExtraInp &inp, ThreadRef &interactor);
};

template <
    std::copyable ExtraInp, ThreadInteractor<ExtraInp> I,
    ThreadCallableForInteractor<ExtraInp, typename I::BlockRef::ThreadRef> F>
struct ThreadInteractorLaunchable {
  [[no_unique_address]] ExtraInp inp;
  [[no_unique_address]] I interactor;
  [[no_unique_address]] F callable;

  struct BlockRef {
    const ExtraInp &inp;
    typename I::BlockRef interactor;
    const F &callable;

    void operator()(const WorkDivision &division, const GridLocationInfo &info,
                    const unsigned block_idx, const unsigned thread_idx) {
      F copy_callable = callable;
      copy_callable(division, info, block_idx, thread_idx, inp,
                    interactor.thread_init(thread_idx));
    }
  };

  BlockRef block_init(const WorkDivision &division, unsigned block_idx) {
    return {
        .inp = inp,
        .interactor = interactor.block_init(division, block_idx, inp),
        .callable = callable,
    };
  }
};

static_assert(Launchable<ThreadInteractorLaunchable<
                  EmptyExtraInp, MockThreadInteractor, MockThreadCallable>>);
} // namespace kernel
