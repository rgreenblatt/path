#pragma once

#include "meta/concepts.h"

#include <algorithm>
#include <thrust/optional.h>

template <typename T> using Optional = thrust::optional<T>;

inline constexpr auto nullopt_value = thrust::nullopt;

template <typename T> constexpr decltype(auto) create_optional(T &&v) {
  return thrust::make_optional(std::forward<T>(v));
}

template <typename T> concept IsOptional = SpecializationOf<T, Optional>;

template <typename T, typename F>
requires std::convertible_to<decltype(std::declval<F>()()),
                             Optional<T>> constexpr Optional<T>
optional_or_else(const Optional<T> &v, const F &f) {
  if (v.has_value()) {
    return v;
  } else {
    return f();
  }
}

template <typename T>
constexpr Optional<T> optional_or(const Optional<T> &v, const Optional<T> &e) {
  return optional_or_else(v, [&]() { return e; });
}

template <typename T, typename F>
requires std::convertible_to<decltype(std::declval<F>()()), T> constexpr T
optional_unwrap_or_else(const Optional<T> &v, const F &f) {
  if (v.has_value()) {
    return *v;
  } else {
    return f();
  }
}

template <typename T>
constexpr Optional<T> optional_unwrap_or(const Optional<T> &v, const T &e) {
  return optional_unwrap_or_else(v, [&]() { return e; });
}

template <typename T, typename F>
constexpr Optional<decltype(std::declval<F>()(std::declval<T>()))>
optional_map(const Optional<T> &v, const F &f) {
  if (v.has_value()) {
    return f(*v);
  } else {
    return nullopt_value;
  }
}

template <typename T, typename F,
          typename Ret = decltype(std::declval<F>()(std::declval<T>()))>
constexpr Ret optional_and_then(const Optional<T> &v, F &&f) {
  if (v.has_value()) {
    return f(*v);
  } else {
    return nullopt_value;
  }
}

template <typename V, typename FFold, typename FBase>
constexpr auto optional_fold(const FFold &, const FBase &f_base,
                             const Optional<V> &first) {
  return optional_map(first, f_base);
}

template <typename V, typename FFold, typename FBase, typename... T>
constexpr auto optional_fold(const FFold &f_fold, const FBase &f_base,
                             const Optional<V> &first,
                             const Optional<T> &...rest) {
  const auto f_rest = optional_fold(f_fold, f_base, rest...);

  const decltype(f_rest) next = optional_and_then(first, [&](const auto &v) {
    if (f_rest.has_value()) {
      // make optional??
      return f_fold(*f_rest, v);
    } else {
      return create_optional(f_base(v));
    }
  });

  return optional_or(next, f_rest);
}

template <typename... T>
constexpr auto optional_min(const Optional<T> &...values) {
  return optional_fold(
      [](const auto &a, const auto &b) {
        return create_optional(std::min(a, b));
      },
      [](const auto &a) { return a; }, values...);
}

template <typename T>
constexpr Optional<T> create_optional(bool condition, const T &v) {
  if (condition) {
    return v;
  } else {
    return nullopt_value;
  }
}
