#pragma once

#include "meta/std_array_specialization.h"

#include <concepts>

template <typename T> struct AllValuesImpl;

// maybe it would be faster to sort?
template <std::equality_comparable T, std::size_t size>
consteval bool all_values_unique(std::array<T, size> values) {
  for (unsigned i = 0; i < values.size(); ++i) {
    for (unsigned j = 0; j < i; ++j) {
      if (values[i] == values[j]) {
        return false;
      }
    }
  }

  return true;
}

template <std::equality_comparable T, std::size_t size>
consteval bool default_in_array(std::array<T, size> values) {
  if constexpr (std::default_initializable<T>) {
    T default_value{};
    for (auto value : values) {
      if (value == default_value) {
        return true;
      }
    }
    return false;
  } else {
    return true;
  }
}

template <typename T>
concept AllValuesEnumerable = requires(const T &t) {
  requires std::equality_comparable<T>;
  { t < t } -> std::convertible_to<bool>; // less

  // having an array of all values is likely not compile time efficient
  // for some types (integers for examples)
  typename AllValuesImpl<T>;
  requires StdArrayOfType<decltype(AllValuesImpl<T>::values), T>;
  requires all_values_unique(AllValuesImpl<T>::values);
  requires default_in_array(AllValuesImpl<T>::values);
};

template <AllValuesEnumerable T>
constexpr auto AllValues = AllValuesImpl<T>::values;
