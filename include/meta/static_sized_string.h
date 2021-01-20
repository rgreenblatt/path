#pragma once

#include "lib/assert.h"

#include <algorithm>
#include <array>
#include <compare>
#include <type_traits>

namespace static_sized_str {
template <typename C> constexpr size_t constexpr_strlen(const C *v) {
  const C *s = v;

  for (; *s; ++s) {
  }

  return (s - v);
}

template <unsigned n> using IdxT = std::integral_constant<unsigned, n>;

template <unsigned size_in, typename C = char> class StaticSizedStr {
public:
  // zero initialize to ensure null termination
  constexpr StaticSizedStr() : data_{} {}

  constexpr StaticSizedStr(const C *v) {
    std::copy(v, v + size_in, data_.begin());
    data_[size_in] = '\0';
  }

  template <const C *v>
  requires(constexpr_strlen(v) == size_in) constexpr StaticSizedStr(
      std::integral_constant<const C *, v>)
      : StaticSizedStr(v) {}

  constexpr StaticSizedStr(std::string_view v) : StaticSizedStr(v.data()) {}

  static constexpr unsigned size() { return size_in; }

  constexpr const C *c_str() const { return data_.data(); }

  constexpr std::string_view as_view() const { return {c_str(), size_in}; }

  template <unsigned other_size>
  constexpr auto operator+(const StaticSizedStr<other_size, C> &other) const {
    debug_assert(data_[size_in] == '\0');
    debug_assert(other.data_[other_size] == '\0');

    StaticSizedStr<size_in + other_size, C> out;
    std::copy(data_.begin(), data_.end() - 1, out.data_.begin());
    std::copy(other.data_.begin(), other.data_.end(),
              out.data_.begin() + size_in);
    debug_assert(out.data_[out.size()] == '\0');

    return out;
  }

  template <unsigned to_remove>
  requires(to_remove <= size_in) constexpr auto remove_prefix() const {
    StaticSizedStr<size_in - to_remove, C> out;
    std::copy(data_.begin() + to_remove, data_.end(), out.data_.begin());
    debug_assert(out.data_[out.size()] == '\0');

    return out;
  }

  template <unsigned to_remove>
  requires(to_remove <=
           size_in) constexpr auto remove_prefix(IdxT<to_remove>) const {
    return remove_prefix<to_remove>();
  }

  template <unsigned to_remove>
  requires(to_remove <= size_in) constexpr auto remove_suffix() const {
    StaticSizedStr<size_in - to_remove, C> out;
    std::copy(data_.begin(), data_.end() - (to_remove + 1), out.data_.begin());
    out.data_[out.size()] = '\0';

    return out;
  }

  template <unsigned to_remove>
  requires(to_remove <=
           size_in) constexpr auto remove_suffix(IdxT<to_remove>) const {
    return remove_suffix<to_remove>();
  }

  template <unsigned n> constexpr auto rep() const {
    StaticSizedStr<n * size_in, C> out;
    for (unsigned i = 0; i < n; ++i) {
      std::copy(data_.begin(), data_.end() - 1,
                out.data_.begin() + i * size_in);
    }
    out.data_[out.size()] = '\0';

    return out;
  }

  template <unsigned n> constexpr auto rep(IdxT<n>) const { return rep<n>(); }

  template <unsigned first_size, unsigned... rest_size>
  constexpr auto join(const StaticSizedStr<first_size, C> &first,
                      const StaticSizedStr<rest_size, C> &...rest) const {
    return (first + ... + (*this + rest));
  }

  template <unsigned n, unsigned joined_size>
  constexpr auto join_n(const StaticSizedStr<joined_size, C> &joined) const {
    if constexpr (n == 0) {
      return StaticSizedStr<0, C>{};
    } else {
      return (joined + *this).template rep<n - 1>() + joined;
    }
  }

  template <unsigned n, unsigned joined_size>
  constexpr auto join_n(IdxT<n>,
                        const StaticSizedStr<joined_size, C> &joined) const {
    return join_n<n>(joined);
  }

  constexpr auto operator<=>(const StaticSizedStr &other) const = default;

private:
  template <unsigned other_size, typename OtherC> friend class StaticSizedStr;

  // private to ensure null termination
  std::array<C, size_in + 1> data_;
};

template <typename T> struct IsStaticSizedStrImpl : std::false_type {};

template <unsigned size>
struct IsStaticSizedStrImpl<StaticSizedStr<size>> : std::true_type {};

template <typename T>
concept IsStaticSizedStr = IsStaticSizedStrImpl<std::decay_t<T>>::value;

template <typename C, const C *v>
StaticSizedStr(std::integral_constant<const C *, v>)
    -> StaticSizedStr<constexpr_strlen(v), C>;

template <unsigned arr_size, typename C>
StaticSizedStr(C const (&)[arr_size]) -> StaticSizedStr<arr_size - 1, C>;

// in namespace because of short name...
namespace short_func {
// consider this as an equivalent to a user defined string literal
template <unsigned arr_size, typename C>
requires(arr_size !=
         0) constexpr StaticSizedStr<arr_size - 1> s(C const (&v)[arr_size]) {
  // must be null terminated...
  debug_assert(v[arr_size - 1] == '\0');

  return {v};
}
} // namespace short_func

} // namespace static_sized_str
