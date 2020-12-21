#pragma once

#include "meta/all_values.h"

#include <iostream>

template <AllValuesEnumerable T>
static constexpr std::size_t get_idx(const T &value) {
  constexpr auto values = AllValues<T>;
  for (unsigned i = 0; i < values.size(); ++i) {
    if (values[i] == value) {
      return i;
    }
  }

  assert(false);
  __builtin_unreachable();
  return 0;
}
