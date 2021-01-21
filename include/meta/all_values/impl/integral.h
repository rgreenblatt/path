#pragma once

#include "meta/all_values/all_values.h"

#include <limits>

template <std::unsigned_integral T> struct AllValuesImpl<T> {
private:
  static constexpr T max = std::numeric_limits<T>::max();

public:
  static constexpr auto values = [] {
    std::array<T, static_cast<std::size_t>(max) + 1> arr;
    for (std::size_t i = 0; i < arr.size(); ++i) {
      arr[i] = i;
    }

    return arr;
  }();
};
