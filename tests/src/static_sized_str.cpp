#include "meta/static_sized_string.h"

#include <string_view>

using namespace static_sized_str;
using namespace short_func;
using namespace std::literals;

template <unsigned size>
constexpr std::string_view to_view(const StaticSizedStr<size> &v) {
  return std::string_view(v.c_str(), v.size());
}

template <unsigned n> constexpr auto idx_v = IdxT<n>{};

static_assert([] {
  constexpr auto empty = s("");

  static_assert(empty.size() == 0);
  static_assert(constexpr_strlen(empty.c_str()) == 0);
  static_assert(to_view(empty) == ""sv);
  static_assert(empty.remove_prefix<0>() == empty);
  static_assert(empty.remove_suffix<0>() == empty);
  static_assert(empty.remove_prefix(idx_v<0>) == empty);
  static_assert(empty.remove_suffix(idx_v<0>) == empty);
  static_assert(empty.rep<0>() == empty);
  static_assert(empty.rep<1>() == empty);
  static_assert(empty.rep(idx_v<0>) == empty);
  static_assert(empty.rep<7>() == empty);
  static_assert(empty.join(s("a")) == s("a"));
  static_assert(empty.join(s("a"), s("b"), s("c")) == s("abc"));
  static_assert(empty.join_n<3>(s("a")) == s("aaa"));
  static_assert(empty.join_n(idx_v<3>, s("a")) == s("aaa"));

  constexpr auto single = s("a");

  static_assert(single.size() == 1);
  static_assert(constexpr_strlen(single.c_str()) == 1);
  static_assert(to_view(single) == "a"sv);
  static_assert(single.remove_prefix<0>() == single);
  static_assert(single.remove_suffix<0>() == single);
  static_assert(single.remove_prefix<1>() == empty);
  static_assert(single.remove_suffix<1>() == empty);
  static_assert(single.remove_prefix(idx_v<1>) == empty);
  static_assert(single.remove_suffix(idx_v<1>) == empty);
  static_assert(single.rep<0>() == empty);
  static_assert(single.rep<1>() == single);
  static_assert(single.rep<4>() == s("aaaa"));
  static_assert(single.join(s("b"), s("c")) == s("bac"));
  static_assert(single.join(s("b")) == s("b"));
  static_assert(single.join_n<3>(s("b")) == s("babab"));

  constexpr auto many = s("a   a bc");

  static_assert(many.size() == 8);
  static_assert(constexpr_strlen(many.c_str()) == 8);
  static_assert(to_view(many) == "a   a bc"sv);
  static_assert(many.remove_prefix<0>() == many);
  static_assert(many.remove_suffix<0>() == many);
  static_assert(many.remove_suffix<2>() == s("a   a "));
  static_assert(many.remove_prefix<1>() == s("   a bc"));
  static_assert(many.remove_prefix<3>() == s(" a bc"));
  static_assert(many.remove_prefix<8>() == empty);
  static_assert(many.remove_suffix<8>() == empty);
  static_assert(many.rep<0>() == empty);
  static_assert(many.rep<1>() == many);
  static_assert(many.rep<4>() == s("a   a bc"
                                   "a   a bc"
                                   "a   a bc"
                                   "a   a bc"));

  constexpr auto other_many = StaticSizedStr("a   a bc");
  static_assert(other_many == many);

  return true;
}());
