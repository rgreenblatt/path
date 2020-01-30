#pragma once

#include "lib/span_convertable.h"

#include <assert.h>
#include <type_traits>

#ifdef NDEBUG
#define DEFAULT_IGNORE true
#else
#define DEFAULT_IGNORE false
#endif

template <typename T, bool ignore_size = DEFAULT_IGNORE> class Span {
#undef DEFAULT_IGNORE
public:
  constexpr Span(T *ptr, std::size_t size) : ptr_(ptr) {
    if constexpr (!ignore_size) {
      size_ = size;
    }
  }

  template <typename V>
  constexpr Span(V &v) : ptr_(SpanConvertable<V>::ptr(v)) {
    if constexpr (!ignore_size) {
      size_ = SpanConvertable<V>::size(v);
    }
  }

  template <typename V>
  constexpr Span(const V &v) : ptr_(SpanConvertable<V>::ptr(v)) {
    if constexpr (!ignore_size) {
      size_ = SpanConvertable<V>::size(v);
    }
  }

  constexpr Span() {}

  constexpr std::size_t size() const { return size_; }

  constexpr bool checkIndex(const std::size_t index) const {
    if constexpr (ignore_size) {
      return true;
    } else {
      return index < size_;
    }
  }

  constexpr T &operator[](const std::size_t index) const {
    assert(checkIndex(index));
    return ptr_[index];
  }

  constexpr T *data() const { return ptr_; }

  constexpr T *begin() const { return ptr_; }

  constexpr T *end() const { return ptr_ + size_; }

private:
  T *ptr_;

  struct NoSize {};

  typename std::conditional<ignore_size, NoSize, std::size_t>::type size_;
};

template <typename T, bool ignore_size>
class SpanConvertable<Span<T, ignore_size>> {
public:
  constexpr static T *ptr(const Span<T, ignore_size> &v) { return v.data(); }

  constexpr static std::size_t size(const Span<T, ignore_size> &v) { return v.size(); }
};

template <typename T> using SpanSized = Span<T, false>;
