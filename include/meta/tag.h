#pragma once

#include "meta/all_values.h"
#include "meta/get_idx.h"

template <AllValuesEnumerable E, unsigned idx_in> struct Tag {
  static constexpr unsigned idx = idx_in;
  static_assert(idx < AllValues<E>.size());
  static constexpr E value = AllValues<E>[idx];
};

// convenience macro for using tags
#define TAG(tag)                                                               \
  Tag<std::decay_t<decltype(tag)>, get_idx(tag)> {}
