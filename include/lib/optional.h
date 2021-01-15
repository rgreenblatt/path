#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "meta/container_concepts.h"
#include "meta/mock.h"
#include "meta/specialization_of.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <utility>

struct NulloptT {};

inline constexpr auto nullopt_value = NulloptT{};

template <typename T> class Optional;

template <typename T>
concept IsOptional = SpecializationOf<T, Optional>;

// Can't use std::optional and thrust::optional takes decades to
// compile so instead we implement optional...
// This was probably a mistake - took longer than expected to test and
// implement. But at least I now have a better understanding of alignment,
// placement new, and exactly how copy/move assigment/construction should
// work for an option...
//
// Note that this optional implements many methods (from rust) which aren't part
// of std::optional
template <typename T> class Optional {
public:
  constexpr Optional() : has_value_(false) {}

  constexpr Optional(const NulloptT &) : Optional() {}

  constexpr Optional(const T &value) requires CopyConstructable<T>
      : has_value_(true) {
    construct_in_place(*this, value);
  }

  constexpr Optional(T &&value) requires MoveConstructable<T>
      : has_value_(true) {
    construct_in_place(*this, std::forward<T>(value));
  }

  constexpr Optional(const Optional &other) requires(
      TriviallyCopyConstructable<T>) = default;

  constexpr Optional(const Optional &other) requires(
      !TriviallyCopyConstructable<T> && CopyConstructable<T>)
      : has_value_(other.has_value_) {
    if (has_value_) {
      construct_in_place(*this, *other);
    }
  }

  constexpr Optional(Optional &&other) requires(TriviallyMoveConstructable<T>) =
      default;

  constexpr Optional(Optional &&other) requires(
      !TriviallyMoveConstructable<T> && MoveConstructable<T>)
      : has_value_(other.has_value_) {
    if (has_value_) {
      construct_in_place(*this, std::move(*other));
      other.has_value_ = false;
    }
  }

  constexpr Optional &operator=(const Optional &other) requires(
      TriviallyCopyAssignable<T>) = default;

  constexpr Optional &
  operator=(const Optional &other) requires(!TriviallyCopyAssignable<T> &&
                                            CopyAssignable<T>) {
    if (this != &other) {
      if (has_value_ && other.has_value_) {
        **this = *other;
      } else if (other.has_value_) {
        construct_in_place(*this, *other);
      } else {
        this->~Optional();
      }

      has_value_ = other.has_value_;
    }

    return *this;
  }

  constexpr Optional &
  operator=(Optional &&other) requires(TriviallyMoveAssignable<T>) = default;

  constexpr Optional &
  operator=(Optional &&other) requires(!TriviallyMoveAssignable<T> &&
                                       MoveAssignable<T>) {
    if (this != &other) {
      if (has_value_ && other.has_value_) {
        **this = std::move(*other);
      } else if (other.has_value_) {
        construct_in_place(*this, std::move(*other));
      } else {
        this->~Optional();
      }

      has_value_ = other.has_value_;
    }

    return *this;
  }

  constexpr ~Optional() requires(TriviallyDestructable<T>) = default;

  constexpr ~Optional() requires(!TriviallyDestructable<T> && Destructable<T>) {
    // destruct
    if (has_value_) {
      value.~T();
    }
  }

  ATTR_PURE_NDEBUG constexpr const T &operator*() const {
    debug_assert(has_value());
    return value;
  }

  ATTR_PURE_NDEBUG constexpr T &operator*() {
    debug_assert(has_value());
    return value;
  }

  constexpr const T *operator->() const { return &(**this); }

  constexpr T *operator->() { return &(**this); }

  ATTR_PURE constexpr bool has_value() const { return has_value_; }

  template <typename F>
  requires std::convertible_to<decltype(std::declval<F>()()), Optional>
  constexpr Optional or_else(const F &f) const {
    if (has_value()) {
      return *this;
    } else {
      return f();
    }
  }

  // or is keyword...
  ATTR_PURE_NDEBUG constexpr Optional op_or(const Optional &o) const {
    return or_else([&]() { return o; });
  }

  template <typename F>
  requires std::convertible_to<decltype(std::declval<F>()()), T>
  constexpr T unwrap_or_else(const F &f) const {
    if (has_value()) {
      return **this;
    } else {
      return f();
    }
  }

  ATTR_PURE_NDEBUG constexpr T unwrap_or(const T &o) const {
    return unwrap_or_else([&]() { return o; });
  }

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

private : template <typename V>
          constexpr static void
          construct_in_place(Optional &cls, V &&v) {
    new (&cls.value) T(std::forward<V>(v));
  }

  struct EmptyT {};

  [[no_unique_address]] union {
    [[no_unique_address]] EmptyT empty;
    [[no_unique_address]] T value;
  };
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

template <typename T>
ATTR_PURE_NDEBUG constexpr Optional<T> create_optional(bool condition,
                                                       const T &v) {
  if (condition) {
    return v;
  } else {
    return nullopt_value;
  }
}
