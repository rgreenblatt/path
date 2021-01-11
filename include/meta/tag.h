#pragma once

#include "meta/all_values.h"
#include "meta/get_idx.h"

// allow for using an index directly
template<unsigned idx_in> struct NTag {
  static constexpr unsigned idx = idx_in;
};

template <AllValuesEnumerable E, unsigned idx_in>
requires(idx_in < AllValues<E>.size()) struct Tag {
  static constexpr unsigned idx = idx_in;
  static constexpr E value = AllValues<E>[idx];

  Tag() = default;

  // implicit conversion
  Tag(NTag<idx_in>) {}
};

// convenience macro for using tags
#define TAG(tag)                                                               \
  Tag<std::decay_t<decltype(tag)>, get_idx(tag)> {}
