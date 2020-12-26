#pragma once

#include "meta/specialization_of.h"
#include "meta/mock.h"

#include <algorithm>
#include <concepts>
#include <utility>
#include <cassert>
#include <cstddef>

struct NulloptT { };

inline constexpr auto nullopt_value = NulloptT{};

template<std::movable T>
class Optional;

template <typename T> concept IsOptional = SpecializationOf<T, Optional>;

// Can't use std::optional and thrust::optional takes decades to
// compile so instead we implement optional...
//
// Note that this optional implements many methods (from rust) which aren't part
// of std::optional
template<std::movable T>
class Optional {
public:
  constexpr Optional() : has_value_(false) {}
  constexpr Optional(const NulloptT &) : Optional() {}
  constexpr Optional(T &&value) : has_value_(true) {
    ::new (reinterpret_cast<void*>(bytes_.data())) T(std::forward<T>(value));
  }
  constexpr Optional(const T &value) : has_value_(true) {
    ::new (reinterpret_cast<void*>(bytes_.data())) T(value);
  }
  constexpr Optional operator=(T &value)  {
    return Optional(std::forward<T>(value));
  }
  constexpr Optional operator=(const T &value)  {
    return Optional(value);
  }

  constexpr const T& operator*() const {
    assert(has_value());
    return *reinterpret_cast<const T*>(bytes_.data());
  }
  constexpr T& operator*() {
    assert(has_value());
    return *reinterpret_cast<T*>(bytes_.data());
  }
  constexpr const T* operator->() const {
    return &(**this);
  }
  constexpr T* operator->() {
    return &(**this);
  }
  
  constexpr bool has_value() const {
    return has_value_;
  }

  template <typename F>
  requires std::convertible_to<decltype(std::declval<F>()()),
                               Optional> constexpr Optional
  or_else(const F &f) const {
    if (has_value()) {
      return *this;
    } else {
      return f();
    }
  }

  // or is keyword...
  constexpr Optional op_or(const Optional &o) const {
    return or_else([&]() { return o; });
  }

  template <typename F>
  requires std::convertible_to<decltype(std::declval<F>()()), T> constexpr T
  unwrap_or_else(const F &f) const {
    if (has_value()) {
      return **this;
    } else {
      return f();
    }
  }

  constexpr T unwrap_or(const T &o) const {
    return unwrap_or_else([&]() { return o; });
  }

  template<typename F>
  using TypeCalledOnT = decltype(std::declval<F>()(std::declval<T>()));

  template <typename F>
  constexpr auto op_map(F &&f) const -> Optional<decltype(f(**this))> {
    if (has_value()) {
      return f(**this);
    } else {
      return nullopt_value;
    }
  }

  template <typename F>
  constexpr auto and_then(F &&f) const 
      -> decltype(f(**this)) requires IsOptional<decltype(f(**this))> {
    if (has_value()) {
      return f(**this);
    } else {
      return nullopt_value;
    }
  }

private:
  std::array<std::byte, sizeof(T)> bytes_;
  bool has_value_;
};

static_assert(IsOptional<Optional<MockMovable>>);

template <typename FFold, typename FBase, typename V>
constexpr auto optional_fold(FFold &&, FBase &&f_base,
                             const Optional<V> &first) {
  return first.op_map(f_base);
}

template <typename FFold, typename FBase, typename V, typename... T>
constexpr auto optional_fold(FFold &&f_fold, FBase &&f_base,
                             const Optional<V> &first,
                             const Optional<T> &...rest) {
  const auto f_rest = optional_fold(std::forward<FFold>(f_fold),
                                    std::forward<FBase>(f_base), rest...);

  const decltype(f_rest) next = first.and_then([&](const auto &v) {
    if (f_rest.has_value()) {
      return f_fold(*f_rest, v);
    } else {
      return Optional(f_base(v));
    }
  });

  return next.op_or(f_rest);
}

template <typename... T>
constexpr auto optional_min(const Optional<T> &...values) {
  return optional_fold(
      [](const auto &a, const auto &b) {
        return Optional(std::min(a, b));
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
