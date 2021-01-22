#pragma once

#include "meta/all_values/all_values.h"
#include "meta/all_values/get_idx.h"

template <AllValuesEnumerable E, unsigned idx_in>
requires(idx_in < AllValues<E>.size()) struct Tag {
  static constexpr unsigned idx = idx_in;
  static constexpr E value = AllValues<E>[idx];

  Tag() = default;

  constexpr operator E() const { return value; }
  constexpr E operator()() const { return value; }
};

template <auto value>
requires(AllValuesEnumerable<std::decay_t<decltype(value)>> &&get_idx(value) <
         AllValues<std::decay_t<decltype(value)>>.size()) using TagT =
    Tag<std::decay_t<decltype(value)>, get_idx(value)>;

template <auto value> constexpr auto tag_v = TagT<value>{};

// convenience macros for using tags (prefer TagT and tag_v when the type is
// structural...)
#define TAGT(tag) Tag<std::decay_t<decltype(tag)>, get_idx(tag)>
#define TAGV(tag) TAGT(tag)()
