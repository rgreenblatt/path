#pragma once

#include "meta/all_values.h"
#include "meta/get_idx.h"
#include "meta/n_tag.h"

// allow for using an index directly

template <AllValuesEnumerable E, unsigned idx_in>
requires(idx_in < AllValues<E>.size()) struct Tag {
  static constexpr unsigned idx = idx_in;
  static constexpr E value = AllValues<E>[idx];

  Tag() = default;

  // implicit conversion
  Tag(NTag<idx_in>) {}
};

template <auto value_in>
requires(AllValuesEnumerable<decltype(value_in)> &&get_idx(value_in) <
         AllValues<decltype(value_in)>.size()) struct TTag {
  static constexpr unsigned idx = get_idx(value_in);
  static constexpr auto value = value_in;

  TTag() = default;

  // implicit conversion - must be defined here because TTag may not
  // exist (if type isn't structural) while Tag always exists...
  operator Tag<decltype(value_in), idx>() const { return {}; }
  TTag(Tag<decltype(value_in), idx>) {}
};

// convenience macros for using tags
#define TAGT(tag) Tag<std::decay_t<decltype(tag)>, get_idx(tag)>
#define TAG(tag) TAGT(tag)()

template <typename T, unsigned idx> Tag<T, idx> to_tag(NTag<idx>) { return {}; }
template <typename T, T v> TAGT(v) to_tag(TTag<v>) { return {}; }

template <typename T, typename E>
concept TagType = requires(T v) {
  requires AllValuesEnumerable<E>;
  to_tag<E>(v);
};
