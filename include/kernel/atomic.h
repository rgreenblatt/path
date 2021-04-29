#pragma once

#include "execution_model/execution_model.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"

#include <atomic>
#include <concepts>

namespace kernel {

namespace atomic {
namespace detail {
// inspired by
// https://codereview.stackexchange.com/questions/113439/copyable-atomic

/**
 * Drop in replacement for std::atomic that provides a copy constructor and copy
 * assignment operator.
 *
 * Contrary to normal atomics, these atomics don't prevent the generation of
 * default constructor and copy operators for classes they are members of.
 *
 * Copying those atomics is thread safe, but be aware that
 * it doesn't provide any form of synchronization.
 */
template <std::copyable T> class CopyableAtomic : public std::atomic<T> {
public:
  using std::atomic<T>::atomic;

  constexpr CopyableAtomic(const CopyableAtomic<T> &other)
      : CopyableAtomic(other.load(std::memory_order_relaxed)) {}

  CopyableAtomic &operator=(const CopyableAtomic<T> &other) {
    this->store(other.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    return *this;
  }
};
} // namespace detail
} // namespace atomic

template <ExecutionModel exec, std::copyable F> class Atomic {
public:
  Atomic() = default;

  Atomic(F value) : inner_{value} {}

  HOST_DEVICE F fetch_add(F to_add) {
    if constexpr (exec == ExecutionModel::GPU) {
      return atomicAdd(&inner_, to_add);
    } else {
      return inner_.fetch_add(to_add);
    }
  }

  // WARNING, this isn't thread safe in any sense
  HOST_DEVICE F as_inner() const {
    if constexpr (exec == ExecutionModel::GPU) {
      return inner_;
    } else {
      return inner_.load();
    }
  }

  HOST_DEVICE F load() const {
    if constexpr (exec == ExecutionModel::GPU) {
#ifdef __CUDA_ARCH__
      // https://stackoverflow.com/questions/32341081/how-to-have-atomic-load-in-cuda

      // volatile to bypass cache
      const volatile F *vaddr = &inner_;

      // for seq_cst loads. Remove for acquire semantics.
      __threadfence();

      const F value = *vaddr;

      // fence to ensure that dependent reads are correctly ordered
      __threadfence();

      return value;
#else
      unreachable_unchecked();
#endif
    } else {
      return inner_.load();
    }
  }

private:
  [[no_unique_address]] std::conditional_t<exec == ExecutionModel::GPU, F,
                                           atomic::detail::CopyableAtomic<F>>
      inner_;
};

static_assert(std::copyable<Atomic<ExecutionModel::CPU, int>>);
static_assert(std::copyable<Atomic<ExecutionModel::GPU, int>>);
} // namespace kernel
