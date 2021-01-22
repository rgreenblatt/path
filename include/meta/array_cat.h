#pragma once

#include "lib/assert.h"

#include <algorithm>
#include <array>

template <typename T, std::size_t... sizes>
static constexpr auto array_cat(const std::array<T, sizes> &...arr) {
  constexpr std::size_t out_size = (... + sizes);
  std::array<T, out_size> out;
  unsigned start = 0;
  ((std::copy(arr.begin(), arr.end(), out.begin() + start),
    start += arr.size()),
   ...);
  debug_assert_assume(start == out_size);

  return out;
}
