#pragma once

#include <algorithm>
#include <array>

template <typename T, std::size_t size, typename F>
constexpr auto array_transform(const std::array<T, size> &in, F &&f) {
  std::array<decltype(f(in[0])), size> out;
  std::transform(in.begin(), in.end(), out.begin(), f);

  return out;
}
