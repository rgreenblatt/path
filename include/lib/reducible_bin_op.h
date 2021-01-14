#pragma once

#include "meta/mock.h"

#include <concepts>

template <typename F, typename T>
concept BinOp = requires(const F &f, const T &l, const T &r) {
  requires std::copyable<T>;
  { f(l, r) } -> std::convertible_to<T>;
};

template <typename T> struct MockBinOp : MockNoRequirements {
  T operator()(const T& l, const T& r) const;
};

static_assert(BinOp<MockBinOp<int>, int>);
