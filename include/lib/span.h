#pragma once

#include "lib/span_convertable.h"

#include <assert.h>
#include <type_traits>

template <typename T, bool is_sized = false> class Span {
public:
  constexpr Span(T *ptr, std::size_t size) : ptr_(ptr) {
    if constexpr (use_size) {
      size_ = size;
    }
  }

  template <typename V>
  constexpr Span(V &v) : ptr_(SpanConvertable<V>::ptr(v)) {
    if constexpr (use_size) {
      size_ = SpanConvertable<V>::size(v);
    }
  }

  template <typename V>
  constexpr Span(const V &v) : ptr_(SpanConvertable<V>::ptr(v)) {
    if constexpr (use_size) {
      size_ = SpanConvertable<V>::size(v);
    }
  }

  constexpr Span() {}

  constexpr std::size_t size() const {
    static_assert(is_sized, "size method can't be used if span isn't sized");
    return size_;
  }

  constexpr bool checkIndex(const std::size_t index) const {
    if constexpr (is_debug) {
      return index < size_;
    } else {
      return true;
    }
  }

  constexpr T &operator[](const std::size_t index) const {
    assert(checkIndex(index));
    return ptr_[index];
  }

  constexpr T *data() const { return ptr_; }

  constexpr T *begin() const { return ptr_; }

  constexpr T *end() const { return ptr_ + size(); }

private:
  T *ptr_;

  struct NoSize {};

  static constexpr bool is_debug =
#ifdef NDEBUG
      false
#else
      true
#endif
      ;
  static constexpr bool use_size = is_sized || is_debug;

  typename std::conditional_t<use_size, std::size_t, NoSize> size_;

  friend class SpanConvertable<Span>;
};

template <typename T, bool is_sized> class SpanConvertable<Span<T, is_sized>> {
public:
  constexpr static T *ptr(const Span<T, is_sized> &v) { return v.ptr_; }

  constexpr static std::size_t size(const Span<T, is_sized> &v) {
    return v.size_;
  }
};

template <typename T> using SpanSized = Span<T, true>;
