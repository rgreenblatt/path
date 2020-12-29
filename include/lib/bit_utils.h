#pragma once

#include "lib/assert.h"

#include <concepts>
#include <limits>
#include <climits>
#include <cstdint>

template <std::integral T>
inline constexpr unsigned bits_per = sizeof(T) * CHAR_BIT;

template<std::unsigned_integral T>
requires (bits_per<T> <= bits_per<unsigned long long>)
inline constexpr uint8_t popcount(T v) {
  if constexpr (bits_per<T> <= bits_per<unsigned>) {
    return __builtin_popcount(v);
  } else if constexpr (bits_per<T> <= bits_per<unsigned long>) {
    return __builtin_popcountl(v);
  } else {
    return __builtin_popcountll(v);
  }
}

// use static_assert for examples and testing
static_assert(popcount(0b1100u) == 2);
static_assert(popcount(0b0u) == 0);
static_assert(popcount(0b11111110u) == 7);

// note that the exact type matters a lot for the behavior of this function
// see examples below
template<std::unsigned_integral T>
requires (bits_per<T> <= bits_per<unsigned long long>)
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

static_assert(count_leading_zeros(static_cast<uint8_t>(0b1u)) == 7);
static_assert(count_leading_zeros(static_cast<uint16_t>(0b1u)) == 15);
static_assert(count_leading_zeros(static_cast<uint16_t>(0b1111u)) == 12);
static_assert(count_leading_zeros(0b1u) == 31);
static_assert(count_leading_zeros(static_cast<uint64_t>(0b1u)) == 63);
static_assert(count_leading_zeros(0b100u) == 29);
static_assert(count_leading_zeros(0b10111u) == 27);

template<std::unsigned_integral T>
constexpr bool power_of_2(T n) {
  return (n > 0 && ((n & (n - 1)) == 0));
};

static_assert(!power_of_2(0b0u));
static_assert(power_of_2(0b1u));
static_assert(power_of_2(0b10u));
static_assert(power_of_2(0b100000u));
static_assert(!power_of_2(0b100100u));
static_assert(!power_of_2(0b111111u));
static_assert(!power_of_2(0b111u));
static_assert(!power_of_2(0b101u));

template<std::unsigned_integral T>
inline constexpr T closest_power_of_2(T n) {
  if (n == 0) {
    return 1;
  }
  unsigned n_leading_zeros = count_leading_zeros(n);
  debug_assert_assume(n_leading_zeros >= 0);
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

static_assert(closest_power_of_2(0b100u) == 0b100u);
static_assert(closest_power_of_2(0b0u) == 0b1u);
static_assert(closest_power_of_2(0b101u) == 0b100u);
static_assert(closest_power_of_2(0b111u) == 0b1000u);
static_assert(closest_power_of_2(std::numeric_limits<unsigned>::max()) ==
              1u << 31);

template <std::unsigned_integral T> constexpr T bit_mask(unsigned bit_idx) {
  return T(1) << bit_idx;
}

static_assert(bit_mask<unsigned>(0) == 0b1);
static_assert(bit_mask<unsigned>(2) == 0b100);
static_assert(bit_mask<unsigned>(7) == 0b10000000);
static_assert(bit_mask<unsigned>(31) == 0b10000000000000000000000000000000u);

template <std::unsigned_integral T> constexpr T up_to_mask(unsigned n) {
  // in binary, 1 (n + 1) times
  // 0: 1
  // 1: 11
  // ...

  // ternary statement required to avoid undefined behavior
  return n >= bits_per<T> - 1 ? std::numeric_limits<T>::max()
                              : bit_mask<T>(n + T(1)) - T(1);
}

static_assert(up_to_mask<unsigned>(0) == 0b1);
static_assert(up_to_mask<unsigned>(1) == 0b11);
static_assert(up_to_mask<unsigned>(7) == 0b11111111);
static_assert(up_to_mask<unsigned>(31) == 0b11111111111111111111111111111111u);
