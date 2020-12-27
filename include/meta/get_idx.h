#pragma once

#include "lib/assert.h"
#include "meta/all_values.h"

template <AllValuesEnumerable T>
static constexpr std::size_t get_idx(const T &value) {
  constexpr auto values = AllValues<T>;
  for (unsigned i = 0; i < values.size(); ++i) {
    if (values[i] == value) {
      return i;
    }
  }

  unreachable_unchecked();
}
