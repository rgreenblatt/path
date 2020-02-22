#pragma once

#include "lib/span.h"

#include <algorithm>

// previous element less, this element greater_equal
template <typename T, bool ignore_increment_guess = false>
constexpr unsigned
binary_search(unsigned start, unsigned end, T target, unsigned guess,
              unsigned start_search_increment, Span<const T> values,
              bool is_increasing = true) {
  if (start == end) {
    return start;
  }

  assert(guess >= start && guess < end);

  auto cmp = [&](const T &v1, const T &v2) {
    if (is_increasing) {
      return v1 < v2;
    } else {
      return v1 > v2;
    }
  };

  auto next_guess = [&] { return (start + end) / 2; };

  if constexpr (ignore_increment_guess) {
    guess = next_guess();
  }

  while (true) {
    if (cmp(values[guess], target)) {
      start = guess;

      if (start == end - 1) {
        break;
      }

      if constexpr (ignore_increment_guess) {
        guess = next_guess();
      } else {
        guess = std::max(int(next_guess()),
                         int(guess) - int(start_search_increment));
      }
    } else {
      end = guess;

      if (start == end) {
        break;
      }

      if constexpr (ignore_increment_guess) {
        guess = next_guess();
      } else {
        guess = std::min(next_guess(), guess + start_search_increment);
      }
    }

    if constexpr (!ignore_increment_guess) {
      start_search_increment =
          std::min(start_search_increment * 2, end - start);
    }
  }

  return end;
}

template <typename T>
constexpr unsigned binary_search(unsigned start, unsigned end, T target,
                                 Span<const T> values,
                                 bool is_increasing = true) {
  return binary_search<T, true>(start, end, target, start, 0, values,
                                is_increasing);
}
