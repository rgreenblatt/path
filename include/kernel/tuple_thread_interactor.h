#pragma once

#include "kernel/thread_interactor.h"
#include "meta/tuple.h"

namespace kernel {
template <typename ExtraInp, ThreadInteractor<ExtraInp>... I>
struct TupleThreadInteractor {
  [[no_unique_address]] MetaTuple<I...> values;

  struct BlockRef {
    [[no_unique_address]] MetaTuple<typename I::BlockRef...> values;
    using ThreadRef = MetaTuple<typename I::BlockRef::ThreadRef...>;

    HOST_DEVICE auto thread_init(unsigned thread_idx) {
      return boost::hana::unpack(values, [&](auto &...values) -> ThreadRef {
        return {values.thread_init(thread_idx)...};
      });
    }
  };

  HOST_DEVICE auto block_init(const WorkDivision &division, unsigned block_idx,
                              const ExtraInp &inp) const {
    return boost::hana::unpack(values, [&](auto &...values) -> BlockRef {
      return {{values.block_init(division, block_idx, inp)...}};
    });
  }
};
} // namespace kernel
