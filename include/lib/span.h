#pragma once

#include <assert.h>
#include <type_traits>

#ifdef NDEBUG
#define DEFAULT_IGNORE true
#else
#define DEFAULT_IGNORE false
#endif

struct NoSize {};

template <typename T, bool ignore_size = DEFAULT_IGNORE> class Span {
public:
  constexpr Span(T *ptr, std::size_t size) : ptr_(ptr) {
    if constexpr (!ignore_size) {
      size_ = size;
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
  typename std::conditional<ignore_size, NoSize, std::size_t>::type size_;
};

#undef DEFAULT_IGNORE
