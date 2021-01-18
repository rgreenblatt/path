#pragma once

#ifndef PRINTF_DBG_MACRO_NO_WARNING
#pragma message("WARNING: the 'printf_dbg.h' header is included")
#endif

#define BOOST_HANA_CONFIG_ENABLE_STRING_UDL

#ifndef PRINTF_DBG_HOST_DEVICE
#ifdef __CUDACC__
#define PRINTF_DBG_HOST_DEVICE __host__ __device__
#else
#define PRINTF_DBG_HOST_DEVICE
#endif
#endif

#include "meta/specialization_of.h"

#include <boost/hana/back.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/integral_constant.hpp>
#include <boost/hana/flatten.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/string.hpp>
#include <boost/hana/tuple.hpp>

#include <string_view>
#include <type_traits>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#define PRINTF_DBG_MACRO_UNIX
#elif defined(_MSC_VER)
#define PRINTF_DBG_MACRO_WINDOWS
#endif

#ifdef PRINTF_DBG_MACRO_UNIX
#include <unistd.h>
#endif

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

namespace printf_dbg {
namespace hana = boost::hana;
using namespace hana::literals;

template <typename Format, typename Vals> struct FormatVals {
  Format f;
  Vals vals;

  constexpr FormatVals(Format f, Vals vals) : f(f), vals(vals) {}

  template <typename OtherFormat, typename OtherVals>
  constexpr auto
  operator+(const FormatVals<OtherFormat, OtherVals> &other) const {
    return printf_dbg::FormatVals(f + other.f, hana::concat(vals, other.vals));
  }
};

template <typename T>
concept IsFormatVals = SpecializationOf<T, FormatVals>;

template <typename Str, typename... T>
constexpr auto s(Str str, const T &...vals) {
  return FormatVals(str, hana::make_tuple(vals...));
}

template <typename Str> constexpr auto join(Str) { return s(""_s); }

template <typename Str, typename First, typename... Rest>
constexpr auto join(Str str, const First &first, const Rest &...rest) {
  return (first + ... + (s(str) + rest));
}

template <typename T> struct FmtImpl;

template <typename T>
concept Formattable = requires(const T &v) {
  typename FmtImpl<std::decay_t<T>>;
  { FmtImpl<std::decay_t<T>>::fmt(v) } -> IsFormatVals;
};

template <Formattable T> constexpr auto fmt_v(const T &v) {
  return FmtImpl<std::decay_t<T>>::fmt(v);
}

template <std::unsigned_integral T> struct FmtImpl<T> {
  static constexpr auto fmt(T val) {
    const auto str = [] {
      if constexpr (sizeof(T) <= sizeof(unsigned char)) {
        return "%hhu"_s;
      } else if constexpr (sizeof(T) <= sizeof(unsigned short)) {
        return "%hu"_s;
      } else if constexpr (sizeof(T) <= sizeof(unsigned)) {
        return "%u"_s;
      } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
        return "%lu"_s;
      } else {
        static_assert(sizeof(T) <= sizeof(unsigned long long),
                      "integral type is too large");
        return "%llu"_s;
      }
    }();

    return s(str, val);
  }
};

template <std::signed_integral T> struct FmtImpl<T> {
  static constexpr auto fmt(T val) {
    const auto str = [] {
      if constexpr (sizeof(T) <= sizeof(char)) {
        return "%hhi"_s;
      } else if constexpr (sizeof(T) <= sizeof(short)) {
        return "%hi"_s;
      } else if constexpr (sizeof(T) <= sizeof(int)) {
        return "%i"_s;
      } else if constexpr (sizeof(T) <= sizeof(long)) {
        return "%li"_s;
      } else {
        static_assert(sizeof(T) <= sizeof(long long),
                      "integral type is too large");
        return "%lli"_s;
      }
    }();

    return s(str, val);
  }
};

template <std::floating_point T> struct FmtImpl<T> {
  static constexpr auto fmt(T val) { return s("%g"_s, val); }
};

template <> struct FmtImpl<bool> {
  static constexpr auto fmt(bool val) {
    return s("%s"_s, val ? "false" : "true");
  }
};

template <> struct FmtImpl<char *> {
  static constexpr auto fmt(const char *val) { return s("%s"_s, val); }
};

template <> struct FmtImpl<const char *> {
  static constexpr auto fmt(const char *val) { return s("%s"_s, val); }
};

template <typename T> struct FmtImpl<T *> {
  static constexpr auto fmt(const T *val) {
    return s("%p"_s, reinterpret_cast<const void *>(val));
  }
};

template <typename T, std::size_t size> struct FmtImpl<std::array<T, size>> {
  static constexpr auto fmt(const std::array<T, size> &val) {
    return s("{"_s) +
           boost::hana::unpack(val,
                               [](const auto &...vals) {
                                 return join(", "_s, fmt_v(vals)...);
                               }) +
           s("}"_s);
  }
};

// No good way to handle this for gpu...
inline constexpr bool is_colorized_output_enabled = true;

namespace detail {
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
constexpr hana::string<base_get_type_name<T>[i]...>
hana_string_impl(std::index_sequence<i...>) {
  return {};
}
} // namespace detail

template <typename T>
inline constexpr auto type_name = detail::hana_string_impl<T>(
    std::make_index_sequence<detail::base_get_type_name<T>.size()>());

namespace detail {
static constexpr const char *const ANSI_EMPTY = "";
static constexpr const char *const ANSI_DEBUG = "\x1b[02m";
// static constexpr const char *const ANSI_WARN = "\x1b[33m";
static constexpr const char *const ANSI_EXPRESSION = "\x1b[36m";
static constexpr const char *const ANSI_VALUE = "\x1b[01m";
static constexpr const char *const ANSI_TYPE = "\x1b[32m";
static constexpr const char *const ANSI_RESET = "\x1b[0m";

// later could become none constexpr?
PRINTF_DBG_HOST_DEVICE const char *ansi(const char *code) {
  if (is_colorized_output_enabled) {
    return code;
  } else {
    return ANSI_EMPTY;
  }
}

template <typename T>
PRINTF_DBG_HOST_DEVICE decltype(auto)
format_item(std::string_view file_name, int line_number, std::string_view func,
            std::string_view var_name, std::string_view var_type,
            const T &val) {
  const long max_file_name_len = 20;
  const long to_remove =
      static_cast<long>(file_name.size()) - max_file_name_len;
  file_name.remove_prefix(std::max(to_remove, 0l));

  FormatVals start("%s%s[%s%s:%d (%s)]%s %s%s%s =\n%s"_s,
                   hana::make_tuple(ansi(ANSI_RESET), ansi(ANSI_DEBUG),
                                    to_remove < 0 ? "" : "..", file_name.data(),
                                    line_number, func.data(), ansi(ANSI_RESET),
                                    ansi(ANSI_EXPRESSION), var_name.data(),
                                    ansi(ANSI_RESET), ansi(ANSI_VALUE)));
  FormatVals type_end("\n%s%s(%s)%s\n"_s,
                      hana::make_tuple(ansi(ANSI_RESET), ansi(ANSI_TYPE),
                                       var_type.data(), ansi(ANSI_RESET)));

  return start + fmt_v(val) + type_end;
}

template <typename... T>
PRINTF_DBG_HOST_DEVICE decltype(auto)
debug_print(std::string_view file_name, int line_number, std::string_view func,
            std::array<std::string_view, sizeof...(T)> var_names,
            std::array<std::string_view, sizeof...(T)> var_types,
            T &&...vals_in) {
  hana::tuple<T &&...> vals{std::forward<T>(vals_in)...};
  auto out =
      hana::unpack(std::make_index_sequence<sizeof...(T)>{}, [&](auto... i) {
        return (... + format_item(file_name, line_number, func, var_names[i],
                                  var_types[i], vals[i]));
      });

  hana::unpack(out.vals, [&](const auto &...format_vals) {
    printf(out.f.c_str(), format_vals...);
  });

  return hana::back(vals);
}

template <typename... T> decltype(auto) identity(T &&...t) {
  return hana::back(hana::tuple<T &&...>{std::forward<T>(t)...});
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

#define PRINTF_DBG_TYPE_NAME(x) printf_dbg::type_name<decltype(x)>.c_str()

#define PRINTF_DBG(...)                                                        \
  printf_dbg::detail::debug_print(                                             \
      __FILE__, __LINE__, __func__,                                            \
      {PRINTF_DBG_MAP(PRINTF_DBG_STRINGIFY, __VA_ARGS__)},                     \
      {PRINTF_DBG_MAP(PRINTF_DBG_TYPE_NAME, __VA_ARGS__)}, __VA_ARGS__)
#else
#define PRINTF_DBG(...) printf_dbg::detail::identity(__VA_ARGS__)
#endif // PRINTF_DBG_MACRO_DISABLE
