#pragma once

#include "meta/less_comparable.h"
#include "meta/std_array_specialization.h"

#include <algorithm>
#include <compare>
#include <concepts>

template <typename T> struct AllValuesImpl;

// needed for sorting
template <typename T>
concept AllValuesCompare = requires(const T &t) {
  requires std::equality_comparable<T>;
  // TODO: hana tuple compare work around...
  requires LessComparable<T>;
};

template <AllValuesCompare T, std::size_t size>
consteval bool all_values_unique(std::array<T, size> values) {
  // arbitrary threshold - haven't done any real benchmarking,
  // I just know that sort is much much faster for large arrays
  constexpr std::size_t sort_threshold = 8;

  if constexpr (size > sort_threshold) {
    std::sort(values.begin(), values.end());
    auto it = std::unique(values.begin(), values.end());

    return it == values.end();
  } else {
    for (unsigned i = 0; i < values.size(); ++i) {
      for (unsigned j = 0; j < i; ++j) {
        if (values[i] == values[j]) {
          return false;
        }
      }
    }

    return true;
  }
}

template <typename T>
concept AllValuesEnumerable = requires(const T &t) {
  requires AllValuesCompare<T>;

  // having an array of all values is likely not compile time efficient
  // for some types (integers for examples)
  typename AllValuesImpl<T>;
  requires StdArrayOfType<decltype(AllValuesImpl<T>::values), T>;
  // TODO: could improve compile times by making this optional...
  requires all_values_unique(AllValuesImpl<T>::values);
};

template <AllValuesEnumerable T>
constexpr auto AllValues = AllValuesImpl<T>::values;

struct EmptyEnumerable {
  constexpr auto operator<=>(const EmptyEnumerable &other) const = default;
};

template <> struct AllValuesImpl<EmptyEnumerable> {
  static constexpr std::array values = {EmptyEnumerable{}};
};
