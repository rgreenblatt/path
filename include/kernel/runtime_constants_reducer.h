#pragma once

#include "execution_model/execution_model.h"
#include "execution_model/host_vector.h"
#include "kernel/thread_interactor.h"
#include "kernel/work_division.h"
#include "lib/bit_utils.h"
#include "lib/optional.h"
#include "lib/reducible_bin_op.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/predicate_for_all_values.h"

// TODO: give this its own namespace
namespace kernel {
namespace detail {
template <typename ItemType, BinOp<ItemType> Op>
inline DEVICE std::optional<ItemType>
runtime_constants_reduce_gpu(ItemType val, const Op &op,
                             unsigned reduction_factor, unsigned block_size,
                             unsigned thread_idx);
} // namespace detail

// commutative operations only!
template <ExecutionModel exec, typename ItemType>
class RuntimeConstantsReducer {
public:
  class BlockRef {
  public:
    class ThreadRef {
    public:
      class ReduceAtIdxRef {
      public:
        template <BinOp<ItemType> Op>
        HOST_DEVICE auto reduce(ItemType val, const Op &op,
                                unsigned reduction_factor) {
          debug_assert_assume(reduction_factor != 0);
          debug_assert_assume(power_of_2(ref_.ref_.block_size_));
          debug_assert_assume(ref_.thread_idx_ < ref_.ref_.block_size_);
          debug_assert_assume(power_of_2(reduction_factor));
          debug_assert_assume(ref_.ref_.block_size_ % reduction_factor == 0);
          debug_assert_assume(reduction_factor <= ref_.ref_.block_size_);

          unsigned reduction_factor_idx = ref_.thread_idx_ % reduction_factor;

          return [&]() -> std::optional<ItemType> {
            if constexpr (exec == ExecutionModel::CPU) {
              auto &item = ref_.ref_.items_[idx_];
              if (reduction_factor_idx == 0) {
                debug_assert(!item.has_value());
                item = val;
              } else {
                debug_assert(item.has_value());
                item = op(*item, val);
              }

              if (reduction_factor_idx == reduction_factor - 1) {
                auto out = std::move(item);
                item = std::nullopt;
                return out;
              } else {
                return std::nullopt;
              }
            } else {
              static_assert(exec == ExecutionModel::GPU);
              return detail::runtime_constants_reduce_gpu(
                  val, op, reduction_factor, ref_.ref_.block_size_,
                  ref_.thread_idx_);
            }
          }();
        }

      private:
        HOST_DEVICE ReduceAtIdxRef(ThreadRef &ref, unsigned idx)
            : ref_(ref), idx_(idx) {}

        friend class ThreadRef;

        ThreadRef &ref_;
        unsigned idx_;
      };

      ATTR_PURE_NDEBUG HOST_DEVICE ReduceAtIdxRef operator[](unsigned idx) {
        return {*this, idx};
      }

    private:
      HOST_DEVICE ThreadRef(BlockRef &ref, unsigned thread_idx)
          : ref_(ref), thread_idx_(thread_idx) {}

      BlockRef &ref_;
      unsigned thread_idx_;

      friend class BlockRef;
    };

    HOST_DEVICE ThreadRef thread_init(unsigned thread_idx) {
      return {*this, thread_idx};
    }

  private:
    HOST_DEVICE BlockRef(unsigned count, unsigned block_size,
                         unsigned block_idx)
        : block_size_(block_size), block_idx_(block_idx) {
      if constexpr (exec == ExecutionModel::CPU) {
        items_.resize(count);
      }
    }

    friend class ThreadRef;
    friend class RuntimeConstantsReducer;

    unsigned block_size_;
    unsigned block_idx_;
    struct EmptyT {};
    [[no_unique_address]] std::conditional_t<
        exec == ExecutionModel::CPU, HostVector<std::optional<ItemType>>,
        EmptyT>
        items_;
  };

  using ThreadRef = typename BlockRef::ThreadRef;

  template <typename Inp>
  HOST_DEVICE BlockRef block_init(const kernel::WorkDivision &division,
                                  unsigned block_idx, const Inp &) const {
    return {count_, division.block_size(), block_idx};
  }

  RuntimeConstantsReducer(unsigned count) : count_(count) {}

private:
  unsigned count_;
};

namespace detail {
template <ExecutionModel exec>
struct IsThreadInteractor
    : std::bool_constant<
          ThreadInteractor<RuntimeConstantsReducer<exec, float>, float>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsThreadInteractor>);
static_assert(ThreadInteractor<
              RuntimeConstantsReducer<ExecutionModel::CPU, float>, float>);
static_assert(ThreadInteractor<
              RuntimeConstantsReducer<ExecutionModel::GPU, float>, float>);
} // namespace detail
} // namespace kernel
