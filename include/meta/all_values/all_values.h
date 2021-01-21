#pragma once

#include "meta/less_comparable.h"
#include "meta/std_array_specialization.h"

#include <algorithm>
#include <compare>
#include <concepts>

template <typename T> struct AllValuesImpl;

template <typename T>
concept AllValuesCompare = requires(const T &t) {
  // TODO: hana tuple compare work around - this should probably just be
  // std::totally_ordered ...
  requires std::equality_comparable<T>;
  requires LessComparable<T>;
};

template <typename T>
concept AllValuesEnumerable = requires(const T &t) {
  requires AllValuesCompare<T>;

  // TODO: having an array of all values is likely not compile time efficient
  // for some types (integers for example)
  typename AllValuesImpl<T>;
  requires StdArrayOfType<decltype(AllValuesImpl<T>::values), T>;
  // TODO: could improve compile times by making this optional...
  // this would be used for impls which can guarantee this property
  // like tuples, ranges, etc...
  requires std::is_sorted(AllValuesImpl<T>::values.begin(),
                          AllValuesImpl<T>::values.end());
  // check unique
  requires(std::adjacent_find(AllValuesImpl<T>::values.begin(),
                              AllValuesImpl<T>::values.end()) ==
           AllValuesImpl<T>::values.end());
};

template <AllValuesEnumerable T>
constexpr auto AllValues = AllValuesImpl<T>::values;

// has only one possible value
struct EmptyEnumerable {
  constexpr auto operator<=>(const EmptyEnumerable &other) const = default;
};

template <> struct AllValuesImpl<EmptyEnumerable> {
  static constexpr std::array values = {EmptyEnumerable{}};
};
