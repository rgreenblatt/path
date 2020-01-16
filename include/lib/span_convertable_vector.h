#pragma once

#include "lib/span_convertable.h"

#include <vector>

template <typename T, typename A>
class SpanConvertable<const std::vector<T, A>> {
public:
  constexpr static const T *ptr(const std::vector<T, A> &v) { return v.data(); }

  constexpr static std::size_t size(const std::vector<T, A> &v) {
    return v.size();
  }
};

template <typename T, typename A> class SpanConvertable<std::vector<T, A>> {
public:
  constexpr static T *ptr(std::vector<T, A> &v) { return v.data(); }

  constexpr static std::size_t size(const std::vector<T, A> &v) {
    return v.size();
  }
};
