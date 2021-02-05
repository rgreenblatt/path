#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"

#include <climits>
#include <concepts>
#include <cstdint>
#include <limits>

// see tests/src/bit_utils.cpp for examples

template <std::integral T>
inline constexpr unsigned bits_per = sizeof(T) * CHAR_BIT;

template <std::unsigned_integral T>
requires(bits_per<T> <= bits_per<unsigned long long>) ATTR_PURE
    inline constexpr uint8_t popcount(T v) {
  if constexpr (bits_per<T> <= bits_per<unsigned>) {
    return __builtin_popcount(v);
  } else if constexpr (bits_per<T> <= bits_per<unsigned long>) {
    return __builtin_popcountl(v);
  } else {
    return __builtin_popcountll(v);
  }
}

// note that the exact type matters a lot for the behavior of this function
// see examples below
template <std::unsigned_integral T>
requires(bits_per<T> <= bits_per<unsigned long long>) ATTR_PURE_NDEBUG
    inline constexpr uint8_t count_leading_zeros(T v) {
  debug_assert_assume(v != 0);
  if constexpr (bits_per<T> <= bits_per<unsigned>) {
    return __builtin_clz(v) - (bits_per<unsigned> - bits_per<T>);
  } else if constexpr (bits_per<T> <= bits_per<unsigned long>) {
    return __builtin_clzl(v) - (bits_per<unsigned long> - bits_per<T>);
  } else {
    return __builtin_clzll(v) - (bits_per<unsigned long long> - bits_per<T>);
  }
}

template <std::unsigned_integral T>
requires(bits_per<T> <= bits_per<unsigned long long>) ATTR_PURE_NDEBUG
    inline constexpr uint8_t log_2_floor(T v) {
  return (bits_per<T> - 1) - count_leading_zeros(v);
}

template <std::unsigned_integral T> ATTR_PURE constexpr bool power_of_2(T n) {
  return popcount(n) == 1;
};

template <std::unsigned_integral T>
ATTR_PURE_NDEBUG inline constexpr T closest_power_of_2(T n) {
  if (n == 0) {
    return 1;
  }
  unsigned n_leading_zeros = count_leading_zeros(n);
  debug_assert_assume(n_leading_zeros <= bits_per<T> - 1);
  unsigned place_of_leading_bit = bits_per<T> - 1 - n_leading_zeros;
  T smaller_power_of_2 = T(1) << place_of_leading_bit;
  debug_assert_assume(smaller_power_of_2 <= n);
  debug_assert_assume(power_of_2(smaller_power_of_2));
  if (place_of_leading_bit == bits_per<T> - 1) {
    return smaller_power_of_2;
  }
  unsigned larger_power_of_2 = smaller_power_of_2 << 1;
  debug_assert_assume(larger_power_of_2 > n);
  debug_assert_assume(power_of_2(larger_power_of_2));
  if (n - smaller_power_of_2 <= larger_power_of_2 - n) {
    return smaller_power_of_2;
  } else {
    return larger_power_of_2;
  }
}

template <std::unsigned_integral T>
ATTR_PURE inline constexpr T bit_mask(unsigned bit_idx) {
  return T(1) << bit_idx;
}

template <std::unsigned_integral T>
ATTR_PURE inline constexpr T up_to_mask(uint8_t n) {
  // in binary, 1 (n + 1) times
  // 0: 1
  // 1: 11
  // ...

  // ternary statement required to avoid undefined behavior
  return n >= bits_per<T> - 1 ? std::numeric_limits<T>::max()
                              : bit_mask<T>(n + T(1)) - T(1);
}
