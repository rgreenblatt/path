#pragma once

// TODO fix headers here
#include <string_view>

// constexpr used to allow function to build targeting device...
// compare to HOST_DEVICE macro

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

template <typename Iter>
constexpr size_t copy_in_n_times(Iter format_iter, std::string_view s,
                                 size_t times) {
  size_t offset = 0;
  for (size_t i = 0; i < times; ++i) {
    copy(s.begin(), s.end(), format_iter);

    offset += s.size();
  }

  return offset;
}

template <typename T> constexpr void swap(T &first, T &second) {
  T temp = first;
  first = second;
  second = temp;
}
