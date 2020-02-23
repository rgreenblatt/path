#pragma once

#include <thrust/optional.h>

#include <algorithm>

// constexpr used to allow function to build targeting device...
// compare to HOST_DEVICE macro

template <typename T, typename F,
          typename Ret = decltype(std::declval<F>()())>
constexpr Ret optional_or_else(const thrust::optional<T> &v, const F &f) {
  if (v.has_value()) {
    return v;
  } else {
    return f();
  }
}

template <typename T>
constexpr thrust::optional<T> optional_or(const thrust::optional<T> &v,
                                          const thrust::optional<T> &e) {
  return optional_or_else(v, [&]() { return e; });
}

template <typename T, typename F,
          typename Ret = decltype(std::declval<F>()(std::declval<T>()))>
constexpr thrust::optional<Ret> optional_map(const thrust::optional<T> &v,
                                             const F &f) {
  if (v.has_value()) {
    return f(*v);
  } else {
    return thrust::nullopt;
  }
}

template <typename V, typename FFold, typename FBase>
constexpr auto optional_fold(const FFold &, const FBase &f_base,
                             const thrust::optional<V> &first) {
  return optional_map(first, f_base);
}

template <typename V, typename FFold, typename FBase, typename... T>
constexpr auto optional_fold(const FFold &f_fold, const FBase &f_base,
                             const thrust::optional<V> &first,
                             const thrust::optional<T> &... rest) {
  const auto f_rest = optional_fold(f_fold, f_base, rest...);

  const decltype(f_rest) next = first.and_then([&](const auto &v) {
    if (f_rest.has_value()) {
      // make optional??
      return f_fold(*f_rest, v);
    } else {
      return thrust::make_optional(f_base(v));
    }
  });

  return optional_or(next, f_rest);
}

template <typename... T>
constexpr auto optional_min(const thrust::optional<T> &... values) {
  return optional_fold(
      [](const auto &a, const auto &b) {
        return thrust::make_optional(std::min(a, b));
      },
      [](const auto &a) { return a; }, values...);
}

template <typename T>
constexpr thrust::optional<T> make_optional(bool condition, const T &v) {
  if (condition) {
    return v;
  } else {
    return thrust::nullopt;
  }
}
