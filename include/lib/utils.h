#pragma once

#include <string_view>

template <class F, class... Args>
constexpr decltype(auto) invoke(F &&f, Args &&... args) {
  return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <class InputIt, class OutputIt>
constexpr OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  while (first != last) {
    *d_first++ = *first++;
  }
  return d_first;
}

template <typename T> constexpr void swap(T &first, T &second) {
  T temp = first;
  first = second;
  second = temp;
}
