#pragma once

#include "lib/assert.h"
#include "lib/span.h"

template <typename GetValue>
inline constexpr unsigned
search_inclusive_gen(const float target, unsigned size, const GetValue &get,
                     const unsigned binary_search_threshold) {
  if (size < binary_search_threshold) {
    for (unsigned i = 0; i < size; ++i) {
      if (get(i) >= target) {
        return i;
      }
    }

    unreachable_unchecked();
  } else {
    // binary search
    // UNIMPLEMENTED...
    unreachable_unchecked();
  }
}

inline constexpr unsigned
search_inclusive(const float target, SpanSized<const float> values,
                 const unsigned binary_search_threshold) {
  return search_inclusive_gen(
      target, values.size(), [&](unsigned i) { return values[i]; },
      binary_search_threshold);
}
