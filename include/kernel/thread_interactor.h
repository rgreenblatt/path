#pragma once

#include "kernel/work_division.h"
#include "meta/mock.h"

namespace kernel {
template <typename T>
concept ThreadInteractorBlockRef = requires(T &mut_ref, unsigned thread_idx) {
  typename T::ThreadRef;
  { mut_ref.thread_init(thread_idx) } -> std::same_as<typename T::ThreadRef>;
};

template <typename T, typename ExtraInp>
concept ThreadInteractor = requires(const WorkDivision &division, const T &t,
                                    unsigned block_idx, const ExtraInp &inp) {
  std::copyable<T>;
  typename T::BlockRef;
  requires ThreadInteractorBlockRef<typename T::BlockRef>;
  {
    t.block_init(division, block_idx, inp)
    } -> std::same_as<typename T::BlockRef>;
};

struct MockThreadInteractor : MockCopyable {
  struct BlockRef : MockMovable {
    struct ThreadRef : MockMovable {};

    ThreadRef thread_init(unsigned thread_idx);
  };

  template <typename ExtraInp>
  BlockRef block_init(const WorkDivision &division, unsigned block_idx,
                      const ExtraInp &inp) const;
};

static_assert(ThreadInteractorBlockRef<MockThreadInteractor::BlockRef>);
static_assert(ThreadInteractor<MockThreadInteractor, MockNoRequirements>);
} // namespace kernel
