#include "lib/info/printf_dbg.h"
#include "lib/info/printf_dbg_eigen.h"

#include <boost/hana/equal.hpp>

#include <string_view>

using namespace printf_dbg::detail;
using namespace printf_dbg;
using namespace std::literals;
using namespace hana::literals;

static_assert([] {
  static_assert(Formattable<bool>);
  static_assert(Formattable<char *>);
  static_assert(Formattable<const char *>);
  static_assert(Formattable<int *>);
  static_assert(Formattable<const int *>);
  static_assert(Formattable<std::array<int, 0>>);
  static_assert(Formattable<std::array<int, 3>>);
  static_assert(fmt_t<float> == float_fmt);
  static_assert(fmt_t<double> == float_fmt);
  static_assert(fmt_t<int> == s("%i"));
  static_assert(fmt_t<short int> == s("%hi"));
  static_assert(fmt_t<unsigned> == s("%u"));
  static_assert(fmt_t<long unsigned> == s("%lu"));
  static_assert(fmt_t<bool> == s("%s"));
  static_assert(fmt_t<std::array<int, 0>> == s("{}"));
  static_assert(fmt_t<std::array<int, 1>> == s("{%i}"));
  static_assert(fmt_t<std::array<int, 3>> == s("{%i, %i, %i}"));
  static_assert(fmt_vals(8) == hana::make_tuple(8));
  static_assert(fmt_vals(std::array{1, 5, 9}) == hana::make_tuple(1, 5, 9));
  static_assert(fmt_vals(true)[0_c] == "true"sv);
  static_assert(fmt_vals(false)[0_c] == "false"sv);
  static_assert(fmt_t<Eigen::Array3f> == s("{") + float_fmt + s("}\n") +
                                             s("{") + float_fmt + s("}\n") +
                                             s("{") + float_fmt + s("}"));
  static_assert(fmt_t<Eigen::Vector3f> == s("{") + float_fmt + s("}\n") +
                                              s("{") + float_fmt + s("}\n") +
                                              s("{") + float_fmt + s("}"));
  static_assert(fmt_t<Eigen::RowVector3f> ==
                s("{") + s(", ").join_n<3>(float_fmt) + s("}"));

  return true;
}());
