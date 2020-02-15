#pragma once

#include <array>

template <typename T, uint32_t max_size> class Stack {
public:
  constexpr Stack() : size_(0) {}

  constexpr void push(const T &v) {
    assert(size_ < max_size);
    arr_[size_] = v;
  }

  constexpr T pop() {
    size_--;
    return arr_[size_];
  }

  constexpr const T &peek() const { return arr_[size_ - 1]; }

  using SizeType = std::conditional_t<
      (max_size > std::numeric_limits<uint16_t>::max()), uint32_t,
      std::conditional_t<(max_size > std::numeric_limits<uint8_t>::max()),
                         uint16_t, uint8_t>>;

  constexpr SizeType size() const { return size_; }

private:
  std::array<T, max_size> arr_;

  SizeType size_;
};
