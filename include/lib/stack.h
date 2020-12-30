#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"

#include <array>
#include <cstdint>
#include <limits>

template <typename T, uint32_t max_size> class Stack {
public:
  constexpr Stack() : size_(0) {}

  constexpr void push(const T &v) {
    debug_assert_assume(size_ < max_size);
    arr_[size_] = v;
    size_++;
  }

  constexpr T pop() {
    debug_assert_assume(size_ != 0);
    size_--;
    return arr_[size_];
  }

  ATTR_PURE_NDEBUG constexpr const T &peek() const {
    debug_assert_assume(size_ != 0);
    return arr_[size_ - 1];
  }

  using SizeType = std::conditional_t<
      (max_size > std::numeric_limits<uint16_t>::max()), uint32_t,
      std::conditional_t<(max_size > std::numeric_limits<uint8_t>::max()),
                         uint16_t, uint8_t>>;

  ATTR_PURE constexpr SizeType size() const { return size_; }

  ATTR_PURE constexpr bool empty() const { return size_ == 0; }

private:
  std::array<T, max_size> arr_;

  SizeType size_;
};
