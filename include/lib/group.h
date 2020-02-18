#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <tuple>

template <typename T>
HOST_DEVICE inline T get_previous(unsigned i, Span<const T> vals) {
  return i == 0 ? T() : vals[i - 1];
}

HOST_DEVICE inline unsigned get_previous(unsigned i,
                                         Span<const unsigned> vals) {
  return get_previous<unsigned>(i, vals);
}

HOST_DEVICE inline unsigned group_size(unsigned i,
                                       Span<const unsigned> groups) {
  return groups[i] - get_previous(i, groups);
}

HOST_DEVICE inline std::tuple<unsigned, unsigned>
group_start_end(unsigned i, Span<const unsigned> groups) {
  return std::make_tuple(get_previous(i, groups), groups[i]);
}
