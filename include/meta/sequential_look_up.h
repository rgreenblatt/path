#pragma once

#include <cassert>
#include <iostream>
#include <type_traits>

template <unsigned size, typename F>
auto sequential_look_up(unsigned index, F &&f) {
  if (index >= size) {
    std::cerr << "Error: index too large for lookup" << std::endl;
    assert(false);
    abort();
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
