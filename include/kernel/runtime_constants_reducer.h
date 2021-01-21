#pragma once

#include "execution_model/execution_model.h"
#include "kernel/thread_interactor.h"
#include "kernel/work_division.h"
#include "lib/bit_utils.h"
#include "lib/optional.h"
#include "lib/reducible_bin_op.h"
#include "meta/all_values_enum.h"
#include "meta/predicate_for_all_values.h"

// TODO: give this its own namespace
namespace kernel {
namespace detail {
template <typename ItemType, BinOp<ItemType> Op>
inline DEVICE Optional<ItemType>
runtime_constants_reduce_gpu(ItemType val, const Op &op,
                             unsigned reduction_factor, unsigned block_size,
                             unsigned thread_idx);
} // namespace detail

// commutative operations only!
template <ExecutionModel exec, typename ItemType>
struct RuntimeConstantsReducer {
  class BlockRef {
  public:
    class ThreadRef {
    public:
      template <BinOp<ItemType> Op>
      HOST_DEVICE auto reduce(ItemType val, const Op &op,
                              unsigned reduction_factor) {
        debug_assert_assume(reduction_factor != 0);
        debug_assert_assume(power_of_2(ref_.block_size_));
        debug_assert_assume(thread_idx_ < ref_.block_size_);
        debug_assert_assume(power_of_2(reduction_factor));
        debug_assert_assume(ref_.block_size_ % reduction_factor == 0);
        debug_assert_assume(reduction_factor <= ref_.block_size_);

        unsigned reduction_factor_idx = thread_idx_ % reduction_factor;

        return [&]() -> Optional<ItemType> {
          if constexpr (exec == ExecutionModel::CPU) {
            if (reduction_factor_idx == 0) {
              debug_assert_assume(!ref_.item_.has_value());
              ref_.item_ = val;
            } else {
              debug_assert_assume(ref_.item_.has_value());
              ref_.item_ = op(*ref_.item_, val);
            }

            if (reduction_factor_idx == reduction_factor - 1) {
              auto out = std::move(ref_.item_);
              ref_.item_ = nullopt_value;
              return out;
            } else {
              return nullopt_value;
            }
          } else {
            static_assert(exec == ExecutionModel::GPU);
            return detail::runtime_constants_reduce_gpu(
                val, op, reduction_factor, ref_.block_size_, thread_idx_);
          }
        }();
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
    HOST_DEVICE BlockRef(unsigned block_size, unsigned block_idx)
        : block_size_(block_size), block_idx_(block_idx) {}

    friend class ThreadRef;
    friend struct RuntimeConstantsReducer;

    unsigned block_size_;
    unsigned block_idx_;
    struct EmptyT {};
    [[no_unique_address]] std::conditional_t<exec == ExecutionModel::CPU,
                                             Optional<ItemType>, EmptyT>
        item_;
  };

  using ThreadRef = typename BlockRef::ThreadRef;

  template <typename Inp>
  HOST_DEVICE BlockRef block_init(const kernel::WorkDivision &division,
                                  unsigned block_idx, const Inp &) const {
    return {division.block_size(), block_idx};
  }
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
