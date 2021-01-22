#pragma once

#include "lib/assert.h"
#include "meta/all_values/n_tag.h"

// this probably isn't a very efficient implementation
template <unsigned size, typename F>
requires(size != 0) constexpr decltype(auto)
    sequential_dispatch(unsigned index, F &&f) {
  if (index >= size) {
    unreachable_unchecked();
  }

  if constexpr (size == 1) {
    return f(NTag<size - 1>{});
  } else {
    if (index == size - 1) {
      return f(NTag<size - 1>{});
    } else {
      return sequential_dispatch<size - 1>(index, f);
    }
  }
}
