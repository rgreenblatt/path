#pragma once

#include "lib/attribute.h"
#include "meta/all_types_same.h"
#include "meta/specialization_of.h"

#include <boost/hana/fold.hpp>
#include <boost/hana/tuple.hpp>

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

// work around clang cuda bug...
#ifdef __CUDACC__
namespace std {
template <typename T> __device__ optional(T)->optional<T>;
} // namespace std
#endif

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
  return optional_or_else(std::move(in), [&]() { return std::move(other); });
}

template <std::movable T, typename F>
requires requires(F &&f) {
  { f() } -> std::convertible_to<T>;
}
constexpr T optional_unwrap_or_else(std::optional<T> in, F &&f) {
  if (in.has_value()) {
    return std::move(*in);
  } else {
    return f();
  }
}

template <std::movable T>
ATTR_NO_DISCARD_PURE constexpr T optional_unwrap_or(std::optional<T> in,
                                                    T default_v) {
  return optional_unwrap_or_else(std::move(in),
                                 [&]() { return std::move(default_v); });
}

template <typename T, typename F>
constexpr auto optional_and_then(std::optional<T> in, F &&f)
    -> decltype(f(std::move(*in))) requires
    IsOptional<decltype(f(std::move(*in)))> &&
    std::movable<decltype(f(std::move(*in)))> {
  if (in.has_value()) {
    return f(std::move(*in));
  } else {
    return std::nullopt;
  }
}

template <typename T, typename F>
requires requires(F &&f, T v) {
  { f(std::move(v)) } -> std::movable;
}
constexpr auto optional_map(std::optional<T> in, F &&f) {
  return optional_and_then(
      std::move(in), [&](T v) { return std::make_optional(f(std::move(v))); });
}

template <typename FFold, std::movable... T>
requires(AllTypesSame<T...> && sizeof...(T) != 0) constexpr auto optional_fold(
    FFold &&f_fold, std::optional<T>... rest)
    -> std::optional<PackElement<0, T...>>
requires requires(PackElement<0, T...> v) {
  { f_fold(std::move(v), std::move(v)) } -> std::same_as<PackElement<0, T...>>;
}
{
  using Type = PackElement<0, T...>;
  return boost::hana::fold_left(
      boost::hana::tuple<std::optional<T>...>{std::move(rest)...},
      [&](std::optional<Type> l, std::optional<Type> r) -> std::optional<Type> {
        if (l.has_value() && r.has_value()) {
          return f_fold(std::move(*l), std::move(*r));
        }
        return optional_or(std::move(l), std::move(r));
      });
}
