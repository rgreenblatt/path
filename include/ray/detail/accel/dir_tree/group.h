#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <tuple>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
HOST_DEVICE inline unsigned get_previous(unsigned i,
                                         Span<const unsigned> vals) {
  return i == 0 ? 0 : vals[i - 1];
}

HOST_DEVICE inline unsigned group_size(unsigned i,
                                       Span<const unsigned> groups) {
  return groups[i] - get_previous(i, groups);
}

HOST_DEVICE inline std::tuple<unsigned, unsigned>
group_start_end(unsigned i, Span<const unsigned> groups) {
  return std::make_tuple(get_previous(i, groups), groups[i]);
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
