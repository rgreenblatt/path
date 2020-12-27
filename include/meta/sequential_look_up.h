#pragma once

#include "lib/assert.h"

#include <type_traits>

template <unsigned size, typename F>
constexpr decltype(auto) sequential_look_up(unsigned index, F &&f) {
  if (index >= size) {
    unreachable_unchecked();
  }

  if constexpr (size == 1) {
    return f(std::integral_constant<unsigned, size - 1>{});
  } else {
    if (index == size - 1) {
      return f(std::integral_constant<unsigned, size - 1>{});
    } else {
      return sequential_look_up<size - 1>(index, f);
    }
  }
}
