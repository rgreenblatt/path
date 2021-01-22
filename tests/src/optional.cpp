#include "lib/optional.h"
#include "intersect/optional_min.h"

#include <gtest/gtest.h>

#include <compare>

// minimal move only struct for testing...
struct T {
  int a;

  constexpr T(int a) : a{a} {}
  constexpr T(const T &) = delete;
  constexpr T(T &&) = default;
  constexpr T &operator=(T &&) = default;
  constexpr auto operator<=>(const T &) const = default;
};

using TOp = std::optional<T>;

static_assert([] {
  static_assert(!optional_or_else(TOp{}, [] { return TOp{}; }).has_value());
  static_assert(!optional_or(TOp{}, TOp{}).has_value());
  static_assert(optional_or(TOp{}, TOp{2}).has_value());
  static_assert(optional_or(TOp{}, TOp{2})->a == 2);
  static_assert(optional_or(TOp{3}, TOp{2})->a == 3);
  static_assert(optional_unwrap_or_else(TOp{}, [] { return T{3}; }).a == 3);
  static_assert(optional_unwrap_or_else(TOp{4}, [] { return T{3}; }).a == 4);
  static_assert(optional_unwrap_or(TOp{4}, T{3}).a == 4);
  static_assert(!optional_map(TOp{}, [](T) { return 0; }).has_value());
  constexpr auto map_out =
      optional_map(TOp{2}, [](T in) { return 2.5 + in.a; });
  static_assert(map_out.has_value());
  static_assert(*map_out == 4.5);
  static_assert(optional_and_then(TOp{2}, [](T in) {
                  return std::optional{2.5 + in.a};
                }).has_value());
  static_assert(
      !optional_fold([](T a, T b) { return T{a.a + b.a}; }, TOp{}).has_value());
  constexpr auto fold_out =
      optional_fold([](T a, T b) { return T{a.a + b.a}; }, TOp{}, TOp{2},
                    TOp{3}, TOp{}, TOp{-2}, TOp{}, TOp{19});
  static_assert(fold_out.has_value());
  static_assert(fold_out->a == 22);
  static_assert(!intersect::optional_min(TOp{}).has_value());
  constexpr auto single_min_value = intersect::optional_min(TOp{2});
  static_assert(single_min_value.has_value());
  static_assert(single_min_value->a == 2);
  constexpr auto min_value = intersect::optional_min(TOp{2}, TOp{3}, TOp{-2});
  static_assert(min_value.has_value());
  static_assert(min_value->a == -2);

  return true;
}());
