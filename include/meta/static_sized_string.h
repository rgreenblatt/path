#pragma once

#include "lib/assert.h"

#include <algorithm>
#include <array>
#include <compare>
#include <type_traits>

constexpr size_t constexpr_strlen(const char *v) {
  const char *s;

  for (s = v; *s; ++s) {
  }

  return (s - v);
}

template <unsigned size_in> class StaticSizedStr {
public:
  template <unsigned other_size>
  constexpr auto operator+(const StaticSizedStr<other_size> &other) const {
    debug_assert(data_[size_in] == '\0');
    debug_assert(other.data_[other_size] == '\0');

    StaticSizedStr<size_in + other_size> out;
    std::copy(data_.begin(), data_.end() - 1, out.data_.begin());
    std::copy(other.data_.begin(), other.data_.end(),
              out.data_.begin() + size_in);
    debug_assert(out.data_[size_in + other_size] == '\0');

    return out;
  }

  static constexpr unsigned size() { return size_in; }

  constexpr StaticSizedStr(const char *v) {
    std::copy(v, v + size_in, data_.begin());
    data_[size_in] = '\0';
  }

  constexpr StaticSizedStr(std::string_view v) {
    std::copy(v.begin(), v.begin() + size_in, data_.begin());
    data_[size_in] = '\0';
  }

  template <const char *v>
  requires(constexpr_strlen(v) == size_in) constexpr StaticSizedStr(
      std::integral_constant<const char *, v>) {
    return StaticSizedStr(v);
  }

  constexpr const char *c_str() const { return data_.data(); }

  constexpr auto operator<=>(const StaticSizedStr &other) const = default;

private:
  template <unsigned other_size> friend class StaticSizedStr;

  constexpr StaticSizedStr() {}

  // private to ensure null termination
  std::array<char, size_in + 1> data_;
};

template <const char *v>
StaticSizedStr(std::integral_constant<const char *, v>)
    -> StaticSizedStr<constexpr_strlen(v)>;

template <unsigned size>
StaticSizedStr(char const (&)[size]) -> StaticSizedStr<size>;

// namespace to avoid potential conflicts with short name
namespace static_sized_str {
template <unsigned arr_size>
requires(arr_size != 0) constexpr StaticSizedStr<arr_size - 1> s_str(
    char const (&v)[arr_size]) {
  // must be null terminated...
  debug_assert(v[arr_size - 1] == '\0');

  return {v};
}
} // namespace static_sized_str
