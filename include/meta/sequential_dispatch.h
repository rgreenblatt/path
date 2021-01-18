#pragma once

#include "lib/assert.h"
#include "meta/n_tag.h"
#include "meta/n_tag_dispatchable.h"

// this probably isn't a very efficient implementation
template <unsigned size, NTagDispatchable<size> F>
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
