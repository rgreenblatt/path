#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <concepts>
#include <tuple>

template <typename T>
    requires std::integral<T> || std::floating_point<T> HOST_DEVICE inline T
                                 get_previous(unsigned i, Span<const T> vals) {
  return i == 0 ? T(0) : vals[i - 1];
}

template <typename T>
    requires std::integral<T> ||
    std::floating_point<T> HOST_DEVICE inline T get_size(unsigned i,
                                                         Span<const T> vals) {
  return vals[i] - get_previous(i, vals);
}

HOST_DEVICE inline unsigned get_previous(unsigned i,
                                         Span<const unsigned> vals) {
  return get_previous<unsigned>(i, vals);
}

HOST_DEVICE inline unsigned group_size(unsigned i,
                                       Span<const unsigned> groups) {
  return get_size<unsigned>(i, groups);
}

HOST_DEVICE inline std::tuple<unsigned, unsigned>
group_start_end(unsigned i, Span<const unsigned> groups) {
  return std::make_tuple(get_previous(i, groups), groups[i]);
}
