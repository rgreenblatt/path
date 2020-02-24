#pragma once

#include <array>
#include <cstdint>
#include <limits>

template <typename T, uint32_t max_size> class Stack {
public:
  constexpr Stack() : size_(0) {}

  constexpr void push(const T &v) {
    assert(size_ < max_size);
    arr_[size_] = v;
    size_++;
  }

  constexpr T pop() {
    assert(size_ != 0);
    size_--;
    return arr_[size_];
  }

  constexpr const T &peek() const {
    assert(size_ != 0);
    return arr_[size_ - 1];
  }

  using SizeType = std::conditional_t<
      (max_size > std::numeric_limits<uint16_t>::max()), uint32_t,
      std::conditional_t<(max_size > std::numeric_limits<uint8_t>::max()),
                         uint16_t, uint8_t>>;

  constexpr SizeType size() const { return size_; }

  constexpr bool empty() const { return size_ == 0; }

private:
  std::array<T, max_size> arr_;

  SizeType size_;
};
