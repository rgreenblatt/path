#pragma once

#define BOOST_HANA_CONFIG_ENABLE_STRING_UDL

#include "lib/bgra_32.h"
#include "lib/cuda/utils.h"
#include "lib/float_rgb.h"

#include <Eigen/Geometry>
#include <boost/hana/back.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/integral_constant.hpp>
#include <boost/hana/flatten.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/string.hpp>
#include <boost/hana/tuple.hpp>

#include <string_view>
#include <type_traits>

namespace printf_dbg {
namespace detail {
namespace ha = boost::hana;
using namespace ha::literals;

// later could be configured/infered - would require global value copied
// to gpu
inline constexpr bool is_colorized_output_enabled() { return true; }

namespace pretty_function {
// Compiler-agnostic version of __PRETTY_FUNCTION__ and constants to
// extract the template argument in `type_name_impl`

#if defined(__clang__)
#define DBG_MACRO_PRETTY_FUNCTION __PRETTY_FUNCTION__
static constexpr size_t PREFIX_LENGTH =
    sizeof("const char *printf_dbg::detail::type_name_impl() [T = ") - 1;
static constexpr size_t SUFFIX_LENGTH = sizeof("]") - 1;
#elif defined(__GNUC__) && !defined(__clang__)
#define DBG_MACRO_PRETTY_FUNCTION __PRETTY_FUNCTION__
static constexpr size_t PREFIX_LENGTH =
    sizeof("const char* printf_dbg::detail::type_name_impl() [with T = ") - 1;
static constexpr size_t SUFFIX_LENGTH = sizeof("]") - 1;
#elif defined(_MSC_VER)
#define DBG_MACRO_PRETTY_FUNCTION __FUNCSIG__
static constexpr size_t PREFIX_LENGTH =
    sizeof("const char *__cdecl printf_dbg::detail::type_name_impl<") - 1;
static constexpr size_t SUFFIX_LENGTH = sizeof(">(void)") - 1;
#else
#error "This compiler is currently not supported by printf_dbg."
#endif
} // namespace pretty_function

template <typename T> constexpr const char *type_name_impl() {
  return DBG_MACRO_PRETTY_FUNCTION;
}

template <typename T>
inline constexpr auto base_get_type_name = []() -> std::string_view {
  namespace pf = pretty_function;

  std::string_view type_name = type_name_impl<T>();
  type_name.remove_prefix(pf::PREFIX_LENGTH);
  type_name.remove_suffix(pf::SUFFIX_LENGTH);

  return type_name;
}();

template <typename T, std::size_t... i>
constexpr ha::string<base_get_type_name<T>[i]...>
hana_string_impl(std::index_sequence<i...>) {
  return {};
}

template <typename T>
inline constexpr auto type_name = hana_string_impl<T>(
    std::make_index_sequence<base_get_type_name<T>.size()>());

template <typename T, std::size_t... i>
constexpr auto rep_impl(T v, std::index_sequence<i...>) {
  return (... + (void(i), v));
}

template <unsigned N, typename T> constexpr auto rep(T v) {
  return rep_impl(v, std::make_index_sequence<N>{});
}

template <typename Format, typename Vals> struct FormatVals {
  Format f;
  Vals vals;

  constexpr FormatVals(Format f, Vals vals) : f(f), vals(vals) {}

  template <typename OtherFormat, typename OtherVals>
  constexpr auto operator+(const FormatVals<OtherFormat, OtherVals> &other) {
    return detail::FormatVals(f + other.f, ha::concat(vals, other.vals));
  }
};

// not ideal imo, some clean up is possible for sure...
template <typename TIn> constexpr auto debug_value(const TIn &val) {
  using T = std::decay_t<TIn>;
  auto comb = [&](auto str, const auto &...vals) {
    return FormatVals(str, ha::make_tuple(vals...));
  };

  if constexpr (std::is_same<T, Eigen::Affine3f>::value) {
    return debug_value(val.matrix());
  } else if constexpr (std::is_same_v<T, Eigen::Matrix3f>) {
    return comb(rep<3>("x: %f, y: %f, z: %f\n"_s), val(0, 0), val(0, 1),
                val(0, 2), val(1, 0), val(1, 1), val(1, 2), val(2, 0),
                val(2, 1), val(2, 2));
  } else if constexpr (std::is_same_v<T, Eigen::Matrix4f>) {
    return comb(rep<4>("x: %f, y: %f, z: %f, w: %f\n"_s), val(0, 0), val(0, 1),
                val(0, 2), val(1, 0), val(1, 1), val(1, 2), val(2, 0),
                val(2, 1), val(2, 2), val(3, 0), val(3, 1), val(3, 2),
                val(3, 3));
  } else if constexpr (std::is_same_v<T, Eigen::Vector3f> ||
                       std::is_same_v<T, FloatRGB>) {
    return comb("x: %f, y: %f, z: %f\n"_s, val.x(), val.y(), val.z());
  } else if constexpr (std::is_same_v<T, BGRA32>) {
    return comb("x: %u, y: %u, z: %u\n"_s, val.x(), val.y(), val.z());
  } else if constexpr (std::is_pointer_v<T>) {
    if constexpr (std::is_same_v<std::decay_t<decltype(*val)> *, char *>) {
      return comb("%s\n"_s, val);
    } else {
      return comb("%p\n"_s, reinterpret_cast<const void *>(val));
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return comb("%u\n"_s, val);
  } else if constexpr (std::is_same_v<T, float>) {
    return comb("%g\n"_s, val);
  } else if constexpr (std::is_same_v<T, unsigned>) {
    return comb("%u\n"_s, val);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return comb("%lu\n"_s, val);
  } else if constexpr (std::is_same_v<T, bool>) {
    return comb("%s\n"_s, val ? "true" : "false");
  } else {
    static_assert(std::is_same_v<T, int>, "type not yet handled");
    return comb("%d\n"_s, val);
  }
}

#ifdef __CUDACC__
extern "C" {
__device__ inline size_t strlen(const char *v) {
  const char *s;

  for (s = v; *s; ++s) {
  }

  return (s - v);
}
}
#endif

static constexpr const char *const ANSI_EMPTY = "";
static constexpr const char *const ANSI_DEBUG = "\x1b[02m";
// static constexpr const char *const ANSI_WARN = "\x1b[33m";
static constexpr const char *const ANSI_EXPRESSION = "\x1b[36m";
static constexpr const char *const ANSI_VALUE = "\x1b[01m";
static constexpr const char *const ANSI_TYPE = "\x1b[32m";
static constexpr const char *const ANSI_RESET = "\x1b[0m";

// later could become none constexpr?
constexpr const char *ansi(const char *code) {
  if (is_colorized_output_enabled()) {
    return code;
  } else {
    return ANSI_EMPTY;
  }
}

template <typename T>
constexpr decltype(auto) format_item(std::string_view file_name,
                                     int line_number, std::string_view func,
                                     std::string_view var_name,
                                     std::string_view var_type, const T &val) {
  const long max_file_name_len = 20;
  const long to_remove =
      static_cast<long>(file_name.size()) - max_file_name_len;
  file_name.remove_prefix(std::max(to_remove, 0l));

  FormatVals start("%s%s[%s%s:%d (%s)]%s %s%s%s =\n%s"_s,
                   ha::make_tuple(ansi(ANSI_RESET), ansi(ANSI_DEBUG),
                                  to_remove < 0 ? "" : "..", file_name.data(),
                                  line_number, func.data(), ansi(ANSI_RESET),
                                  ansi(ANSI_EXPRESSION), var_name.data(),
                                  ansi(ANSI_RESET), ansi(ANSI_VALUE)));
  FormatVals type_end("%s%s(%s)%s\n"_s,
                      ha::make_tuple(ansi(ANSI_RESET), ansi(ANSI_TYPE),
                                     var_type.data(), ansi(ANSI_RESET)));

  return start + debug_value(val) + type_end;
}

template <typename... T>
HOST_DEVICE decltype(auto)
debug_print(std::string_view file_name, int line_number, std::string_view func,
            std::array<std::string_view, sizeof...(T)> var_names,
            std::array<std::string_view, sizeof...(T)> var_types,
            T &&...vals_in) {
  ha::tuple<T &&...> vals{std::forward<T>(vals_in)...};
  auto out =
      ha::unpack(std::make_index_sequence<sizeof...(T)>{}, [&](auto... i) {
        return (... + format_item(file_name, line_number, func, var_names[i],
                                  var_types[i], vals[i]));
      });

  ha::unpack(out.vals, [&](const auto &...format_vals) {
    printf(out.f.c_str(), format_vals...);
  });

  return ha::back(vals);
}

template <typename... T> decltype(auto) identity(T &&...t) {
  return ha::back(ha::tuple<T &&...>{std::forward<T>(t)...});
}
} // namespace detail
} // namespace printf_dbg

// this code is macro nonsense stolen from dbg.h
#ifndef PRINTF_DBG_MACRO_DISABLE
// Force expanding argument with commas for MSVC, ref:
// https://stackoverflow.com/questions/35210637/macro-expansion-argument-with-commas
// Note that "args" should be a tuple with parentheses, such as "(e1, e2, ...)".
#define PRINTF_DBG_IDENTITY(x) x
#define PRINTF_DBG_CALL(fn, args) PRINTF_DBG_IDENTITY(fn args)

#define PRINTF_DBG_CAT_IMPL(_1, _2) _1##_2
#define PRINTF_DBG_CAT(_1, _2) PRINTF_DBG_CAT_IMPL(_1, _2)

#define PRINTF_DBG_16TH_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11,     \
                             _12, _13, _14, _15, _16, ...)                     \
  _16
#define PRINTF_DBG_16TH(args) PRINTF_DBG_CALL(PRINTF_DBG_16TH_IMPL, args)
#define PRINTF_DBG_NARG(...)                                                   \
  PRINTF_DBG_16TH(                                                             \
      (__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

// PRINTF_DBG_VARIADIC_CALL(fn, data, e1, e2, ...) => fn_N(data, (e1, e2, ...))
#define PRINTF_DBG_VARIADIC_CALL(fn, data, ...)                                \
  PRINTF_DBG_CAT(fn##_, PRINTF_DBG_NARG(__VA_ARGS__))(data, (__VA_ARGS__))

// (e1, e2, e3, ...) => e1
#define PRINTF_DBG_HEAD_IMPL(_1, ...) _1
#define PRINTF_DBG_HEAD(args) PRINTF_DBG_CALL(PRINTF_DBG_HEAD_IMPL, args)

// (e1, e2, e3, ...) => (e2, e3, ...)
#define PRINTF_DBG_TAIL_IMPL(_1, ...) (__VA_ARGS__)
#define PRINTF_DBG_TAIL(args) PRINTF_DBG_CALL(PRINTF_DBG_TAIL_IMPL, args)

#define PRINTF_DBG_MAP_1(fn, args) PRINTF_DBG_CALL(fn, args)
#define PRINTF_DBG_MAP_2(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_1(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_3(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_2(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_4(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_3(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_5(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_4(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_6(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_5(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_7(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_6(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_8(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_7(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_9(fn, args)                                             \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_8(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_10(fn, args)                                            \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_9(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_11(fn, args)                                            \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_10(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_12(fn, args)                                            \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_11(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_13(fn, args)                                            \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_12(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_14(fn, args)                                            \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_13(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_15(fn, args)                                            \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_14(fn, PRINTF_DBG_TAIL(args))
#define PRINTF_DBG_MAP_16(fn, args)                                            \
  fn(PRINTF_DBG_HEAD(args)), PRINTF_DBG_MAP_15(fn, PRINTF_DBG_TAIL(args))

// PRINTF_DBG_MAP(fn, e1, e2, e3, ...) => fn(e1), fn(e2), fn(e3), ...
#define PRINTF_DBG_MAP(fn, ...)                                                \
  PRINTF_DBG_VARIADIC_CALL(PRINTF_DBG_MAP, fn, __VA_ARGS__)

#define PRINTF_DBG_STRINGIFY_IMPL(x) #x
#define PRINTF_DBG_STRINGIFY(x) PRINTF_DBG_STRINGIFY_IMPL(x)

#define PRINTF_DBG_TYPE_NAME(x)                                                \
  printf_dbg::detail::type_name<decltype(x)>.c_str()

#define PRINTF_DBG(...)                                                        \
  printf_dbg::detail::debug_print(                                             \
      __FILE__, __LINE__, __func__,                                            \
      {PRINTF_DBG_MAP(PRINTF_DBG_STRINGIFY, __VA_ARGS__)},                     \
      {PRINTF_DBG_MAP(PRINTF_DBG_TYPE_NAME, __VA_ARGS__)}, __VA_ARGS__)
#else
#define PRINTF_DBG(...) printf_dbg::detail::identity(__VA_ARGS__)
#endif // PRINTF_DBG_MACRO_DISABLE
