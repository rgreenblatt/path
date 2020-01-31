#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <tuple>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
HOST_DEVICE static inline unsigned group_size(unsigned i,
                                              Span<unsigned> group) {
  if (i == 0) {
    return group[i];
  }
  return group[i] - group[i - 1];
}

HOST_DEVICE static inline std::tuple<unsigned, unsigned>
group_start_end(unsigned i, Span<unsigned> group) {
  return std::make_tuple(i == 0 ? 0 : group[i - 1], group[i]);
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
