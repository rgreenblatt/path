#pragma once

#include "kernel/runtime_constants_reducer.h"
#include "lib/cuda/reduce.cuh"

namespace kernel {
namespace detail {
template <typename ItemType, BinOp<ItemType> Op>
inline DEVICE std::optional<ItemType>
runtime_constants_reduce_gpu(ItemType val, const Op &op,
                             unsigned reduction_factor, unsigned block_size,
                             unsigned thread_idx) {
  auto item =
      sub_block_reduce(val, op, thread_idx, block_size, reduction_factor);
  if (thread_idx % reduction_factor == 0) {
    return item;
  } else {
    return std::nullopt;
  }
}
} // namespace detail
} // namespace kernel
