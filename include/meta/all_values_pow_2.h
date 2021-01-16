#pragma once

#include "lib/bit_utils.h"
#include "meta/all_values.h"

// inclusive
template <std::unsigned_integral T, T min, T max>
requires(power_of_2(min) && power_of_2(max) && min <= max) struct Pow2Gen {
  T value;
  constexpr Pow2Gen() : value{min} {}
  constexpr Pow2Gen(T value) : value{value} {
    debug_assert(value <= max);
    debug_assert(power_of_2(value));
  }
  constexpr operator T() const { return value; }
  constexpr operator T &() { return value; }
};

template <unsigned min, unsigned max> using Pow2 = Pow2Gen<unsigned, min, max>;

template <unsigned max> using Pow2InclusiveUpTo = Pow2<1, max>;

template <std::unsigned_integral T, T min, T max>
struct AllValuesImpl<Pow2Gen<T, min, max>> {
  static constexpr auto values = [] {
    std::array<Pow2Gen<T, min, max>, log_2_floor(max) - log_2_floor(min) + 1>
        arr;
    for (T i = 0; i < arr.size(); ++i) {
      arr[i] = min << i;
    }

    return arr;
  }();
};
