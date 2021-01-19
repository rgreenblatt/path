#pragma once

#include "lib/assert.h"
#include "meta/all_values.h"

template <std::unsigned_integral T, T begin, T end>
requires(begin <= end) struct RangeGen {
  T value;
  constexpr RangeGen() requires(begin != end) : value{begin} {}
  constexpr RangeGen(T value) requires(begin != end) : value{value} {
    debug_assert(value >= begin);
    debug_assert(value < end);
  }
  constexpr operator T() const { return value; }
  constexpr T operator()() const { return value; }
};

template <unsigned begin, unsigned end>
using Range = RangeGen<unsigned, begin, end>;
template <std::unsigned_integral T, T end> using UpToGen = RangeGen<T, 0, end>;
template <unsigned end> using UpTo = UpToGen<unsigned, end>;

template <std::unsigned_integral T, T begin, T end>
struct AllValuesImpl<RangeGen<T, begin, end>> {
  static constexpr auto values = [] {
    std::array<RangeGen<T, begin, end>, end - begin> arr;
    if constexpr (begin != end) {
      for (T i = 0; i < arr.size(); ++i) {
        arr[i] = i + begin;
      }
    }

    return arr;
  }();
};
