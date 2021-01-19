#pragma once

#ifndef PRINTF_DBG_MACRO_NO_WARNING
#pragma message("WARNING: the 'printf_dbg.h' header is included")
#endif

#ifndef PRINTF_DBG_HOST_DEVICE
#ifdef __CUDACC__
#define PRINTF_DBG_HOST_DEVICE __host__ __device__
#else
#define PRINTF_DBG_HOST_DEVICE
#endif
#endif

#include "meta/macro_map.h"
#include "meta/specialization_of.h"
#include "meta/static_sized_string.h"

#include <boost/hana/back.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/integral_constant.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/tuple.hpp>

#include <cstdio>
#include <string_view>
#include <type_traits>
#include <utility>

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
__device__ inline size_t strlen(const char *v) { return constexpr_strlen(v); }
}
#endif

namespace printf_dbg {
namespace hana = boost::hana;
using namespace static_sized_str;

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

template <typename Str> constexpr auto join(Str) { return s(s_str("")); }

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
        return s_str("%hhu");
      } else if constexpr (sizeof(T) <= sizeof(unsigned short)) {
        return s_str("%hu");
      } else if constexpr (sizeof(T) <= sizeof(unsigned)) {
        return s_str("%u");
      } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
        return s_str("%lu");
      } else {
        static_assert(sizeof(T) <= sizeof(unsigned long long),
                      "integral type is too large");
        return s_str("%llu");
      }
    }();

    return s(str, val);
  }
};

template <std::signed_integral T> struct FmtImpl<T> {
  static constexpr auto fmt(T val) {
    const auto str = [] {
      if constexpr (sizeof(T) <= sizeof(char)) {
        return s_str("%hhi");
      } else if constexpr (sizeof(T) <= sizeof(short)) {
        return s_str("%hi");
      } else if constexpr (sizeof(T) <= sizeof(int)) {
        return s_str("%i");
      } else if constexpr (sizeof(T) <= sizeof(long)) {
        return s_str("%li");
      } else {
        static_assert(sizeof(T) <= sizeof(long long),
                      "integral type is too large");
        return s_str("%lli");
      }
    }();

    return s(str, val);
  }
};

template <std::floating_point T> struct FmtImpl<T> {
  static constexpr auto fmt(T val) { return s(s_str("%g"), val); }
};

template <> struct FmtImpl<bool> {
  static constexpr auto fmt(bool val) {
    return s(s_str("%s"), val ? "false" : "true");
  }
};

template <> struct FmtImpl<char *> {
  static constexpr auto fmt(const char *val) { return s(s_str("%s"), val); }
};

template <> struct FmtImpl<const char *> {
  static constexpr auto fmt(const char *val) { return s(s_str("%s"), val); }
};

template <typename T> struct FmtImpl<T *> {
  static constexpr auto fmt(const T *val) {
    return s(s_str("%p"), reinterpret_cast<const void *>(val));
  }
};

template <typename T, std::size_t size> struct FmtImpl<std::array<T, size>> {
  static constexpr auto fmt(const std::array<T, size> &val) {
    return s(s_str("{")) +
           boost::hana::unpack(val,
                               [](const auto &...vals) {
                                 return join(s_str(", "), fmt_v(vals)...);
                               }) +
           s(s_str("}"));
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
} // namespace detail

template <typename T>
inline constexpr auto
    type_name = StaticSizedStr<detail::base_get_type_name<T>.size()>(
        detail::base_get_type_name<T>);

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

  FormatVals start(s_str("%s%s[%s%s:%d (%s)]%s %s%s%s =\n%s"),
                   hana::make_tuple(ansi(ANSI_RESET), ansi(ANSI_DEBUG),
                                    to_remove < 0 ? "" : "..", file_name.data(),
                                    line_number, func.data(), ansi(ANSI_RESET),
                                    ansi(ANSI_EXPRESSION), var_name.data(),
                                    ansi(ANSI_RESET), ansi(ANSI_VALUE)));
  FormatVals type_end(s_str("\n%s%s(%s)%s\n"),
                      hana::make_tuple(ansi(ANSI_RESET), ansi(ANSI_TYPE),
                                       var_type.data(), ansi(ANSI_RESET)));

  return start + fmt_v(val) + type_end;
}

// here string_view must be null terminated (for usage with printf...)
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

#ifndef PRINTF_DBG_MACRO_DISABLE
#define PRINTF_DBG_TYPE_NAME(x) printf_dbg::type_name<decltype(x)>.c_str()

#define PRINTF_DBG(...)                                                        \
  printf_dbg::detail::debug_print(                                             \
      __FILE__, __LINE__, __func__, {MACRO_MAP_STRINGIFY_COMMA(__VA_ARGS__)},  \
      {MACRO_MAP_PP_COMMA_MAP(PRINTF_DBG_TYPE_NAME, __VA_ARGS__)},             \
      __VA_ARGS__)
#else
#define PRINTF_DBG(...) printf_dbg::detail::identity(__VA_ARGS__)
#endif // PRINTF_DBG_MACRO_DISABLE
