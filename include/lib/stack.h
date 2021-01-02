#pragma once

#include "array_vec.h"
#include "lib/assert.h"
#include "lib/attribute.h"

#include <concepts>

template <std::semiregular T, unsigned max_size> class Stack {
public:
  constexpr void push(const T &v) { vec_.push_back(v); }

  constexpr T pop() {
    debug_assert_assume(vec_.size() != 0);
    T out = vec_[last_idx()];
    vec_.resize(last_idx());
    return out;
  }

  ATTR_PURE_NDEBUG constexpr const T &peek() const {
    debug_assert_assume(vec_.size() != 0);
    return vec_[last_idx()];
  }

  ATTR_PURE_NDEBUG constexpr auto size() const { return vec_.size(); }

  ATTR_PURE_NDEBUG constexpr bool empty() const { return vec_.empty(); }

private:
  ATTR_PURE_NDEBUG constexpr unsigned last_idx() const {
    return vec_.size() - 1;
  }

  ArrayVec<T, max_size> vec_;
};
