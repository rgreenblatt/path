#pragma once

#include "meta/all_values/all_values.h"
#include "meta/all_values/get_idx.h"
#include "meta/all_values/n_tag.h"

// allow for using an index directly

template <AllValuesEnumerable E, unsigned idx_in>
requires(idx_in < AllValues<E>.size()) struct Tag {
  static constexpr unsigned idx = idx_in;
  static constexpr E value = AllValues<E>[idx];

  Tag() = default;

  // implicit conversion
  Tag(NTag<idx_in>) {}
};

template <auto value>
requires(AllValuesEnumerable<std::decay_t<decltype(value)>> &&get_idx(value) <
         AllValues<std::decay_t<decltype(value)>>.size()) using TagT =
    Tag<std::decay_t<decltype(value)>, get_idx(value)>;

template <auto value> constexpr auto TagV = TagT<value>{};

// convenience macros for using tags (prefer TagT and TagV when the type is
// structural...)
#define TAGT(tag) Tag<std::decay_t<decltype(tag)>, get_idx(tag)>
#define TAGV(tag) TAGT(tag)()

template <typename T, unsigned idx> Tag<T, idx> to_tag(NTag<idx>) { return {}; }
