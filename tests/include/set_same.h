#pragma once

#include <algorithm>
#include <array>
#include <concepts>

template <typename T, std::size_t size, typename O>
requires(std::convertible_to<O, T> ||
         size == 0) constexpr bool set_same_convert(std::array<T, size> l,
                                                    std::array<O, size> r) {
  std::sort(l.begin(), l.end());
  std::sort(r.begin(), r.end());

  if constexpr (size != 0) {
    for (unsigned i = 0; i < size; ++i) {
      if (l[i] != T(r[i])) {
        return false;
      }
    }
  }

  return true;
}

template <typename T, std::size_t size>
constexpr bool set_same(std::array<T, size> l, std::array<T, size> r) {
  return set_same_convert(l, r);
}
