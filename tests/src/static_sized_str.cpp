#include "meta/static_sized_string.h"

#include <string_view>

using namespace static_sized_str;
using namespace std::literals;

template <unsigned size>
constexpr std::string_view to_view(const StaticSizedStr<size> &v) {
  return std::string_view(v.c_str(), v.size());
}

static_assert([] {
  constexpr auto empty = s_str("");

  static_assert(empty.size() == 0);
  static_assert(constexpr_strlen(empty.c_str()) == 0);
  static_assert(to_view(empty) == ""sv);

  constexpr auto single = s_str("a");

  static_assert(single.size() == 1);
  static_assert(constexpr_strlen(single.c_str()) == 1);
  static_assert(to_view(single) == "a"sv);

  constexpr auto many = s_str("a   a bc");

  static_assert(many.size() == 8);
  static_assert(constexpr_strlen(many.c_str()) == 8);
  static_assert(to_view(many) == "a   a bc"sv);

  return true;
}());
