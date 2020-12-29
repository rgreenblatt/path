#pragma once

template<typename T>
constexpr decltype(auto) tone_map(const T& v) {
  return v / (v + 1);
}
