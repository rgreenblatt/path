#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "lib/start_end.h"

#include <concepts>

template <std::copyable T>
requires std::integral<T> || std::floating_point<T>
    ATTR_PURE_NDEBUG HOST_DEVICE inline T
    edges_get_previous(unsigned i, Span<const T> vals) {
  return i == 0 ? T(0) : vals[i - 1];
}

template <std::copyable T>
requires std::integral<T> || std::floating_point<T>
    ATTR_PURE_NDEBUG HOST_DEVICE inline StartEnd<T>
    edges_start_end(unsigned i, Span<const T> vals) {
  return {edges_get_previous(i, vals), vals[i]};
}

template <std::copyable T>
requires std::integral<T> || std::floating_point<T>
    ATTR_PURE_NDEBUG HOST_DEVICE inline T edges_get_size(unsigned i,
                                                         Span<const T> vals) {
  auto [start, end] = edges_start_end(i, vals);
  return end - start;
}
