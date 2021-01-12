#include "lib/span.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

template <bool sized, bool is_debug> constexpr void check_type() {
  Span<int, sized, is_debug> sp;
  Span other = sp;
  static_assert(std::same_as<decltype(other), decltype(sp)>);
}

TEST(Span, type_deduction) {
  const std::vector<int> vec;
  std::vector<int> mut_vec;
  const std::array<int, 1> arr = {1};
  std::array<int, 1> mut_arr;
  Span sp = vec;
  Span mut_sp = mut_vec;
  Span arr_sp = arr;
  Span arr_mut_sp = mut_arr;

  static_assert(std::same_as<decltype(sp), Span<const int, true>>);
  static_assert(std::same_as<decltype(mut_sp), Span<int, true>>);
  static_assert(std::same_as<decltype(arr_sp), Span<const int, true>>);
  static_assert(std::same_as<decltype(arr_mut_sp), Span<int, true>>);

  check_type<false, false>();
  check_type<false, true>();
  check_type<true, false>();
  check_type<true, true>();
}

TEST(Span, correct_size) {
  const std::vector<int> vec(7);
  std::array<int, 4> arr;
  Span vec_sp = vec;
  Span arr_sp = arr;
  EXPECT_EQ(vec_sp.size(), vec.size());
  EXPECT_EQ(arr_sp.size(), arr.size());
}
