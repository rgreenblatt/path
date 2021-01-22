#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "meta/mock.h"
#include "meta/specialization_of.h"

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

inline constexpr auto nullopt_value = std::nullopt;

namespace std {
#ifdef __CUDACC__
template <typename T> __device__ optional(T)->optional<T>;
#endif
} // namespace std

template <typename T>
concept IsOptional = SpecializationOf<T, std::optional>;

// here we implement useful helper methods (mostly from rust...)

template <std::movable T, typename F>
requires requires(F &&f) {
  { f() } -> std::convertible_to<std::optional<T>>;
}
constexpr std::optional<T> optional_or_else(std::optional<T> in, F &&f) {
  if (in.has_value()) {
    return in;
  } else {
    return f();
  }
}

template <std::movable T>
ATTR_NO_DISCARD_PURE constexpr std::optional<T>
optional_or(std::optional<T> in, std::optional<T> other) {
  return optional_or_else(in, [&]() { return other; });
}

template <std::movable T, typename F>
requires requires(F &&f) {
  { f() } -> std::convertible_to<T>;
}
constexpr T optional_unwrap_or_else(std::optional<T> in, F &&f) {
  if (in.has_value()) {
    return *in;
  } else {
    return f();
  }
}

template <std::movable T>
ATTR_NO_DISCARD_PURE constexpr T optional_unwrap_or(std::optional<T> in,
                                                    T default_v) {
  return in.unwrap_or_else([&]() { return default_v; });
}

template <typename T, typename F>
requires requires(F &&f, const T &v) {
  { f(v) } -> std::movable;
}
constexpr auto optional_map(const std::optional<T> &in, F &&f)
    -> std::optional<decltype(f(*in))> {
  if (in.has_value()) {
    return f(*in);
  } else {
    return nullopt_value;
  }
}

template <typename T, typename F>
constexpr auto optional_and_then(const std::optional<T> &in, F &&f)
    -> decltype(f(*in)) requires IsOptional<decltype(f(*in))> {
  if (in.has_value()) {
    return f(*in);
  } else {
    return nullopt_value;
  }
}

template <typename FFold, typename FBase, typename V>
constexpr auto optional_fold(FFold &&, FBase &&f_base,
                             const std::optional<V> &first) {
  return optional_map(first, f_base);
}

template <typename FFold, typename FBase, typename V, typename... T>
constexpr auto optional_fold(FFold &&f_fold, FBase &&f_base,
                             const std::optional<V> &first,
                             const std::optional<T> &...rest) {
  const auto f_rest = optional_fold(std::forward<FFold>(f_fold),
                                    std::forward<FBase>(f_base), rest...);

  const decltype(f_rest) next = optional_and_then(first, [&](const auto &v) {
    if (f_rest.has_value()) {
      return f_fold(*f_rest, v);
    } else {
      return std::optional(f_base(v));
    }
  });

  return optional_or(next, f_rest);
}

template <std::movable T>
ATTR_NO_DISCARD_PURE constexpr std::optional<T> create_optional(bool condition,
                                                                T v) {
  if (condition) {
    return v;
  } else {
    return nullopt_value;
  }
}
