#pragma once

#include "lib/optional.h"
#include "lib/reducible_bin_op.h"

#include <concepts>

namespace kernel {
template <typename T, typename ItemType>
concept Reducer = requires(T &reducer, ItemType item,
                           const MockBinOp<ItemType> &bin_op) {
  requires std::semiregular<ItemType>;
  { reducer.reduce(item, bin_op) } -> std::same_as<Optional<ItemType>>;
};

} // namespace kernel
