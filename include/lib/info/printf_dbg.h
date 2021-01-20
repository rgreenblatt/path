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
#include "meta/pack_element.h"
#include "meta/specialization_of.h"
#include "meta/static_sized_string.h"

#include <boost/hana/back.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/integral_constant.hpp>
#include <boost/hana/flatten.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/size.hpp>
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
__device__ inline size_t strlen(const char *v) {
  return static_sized_str::constexpr_strlen(v);
}
}
#endif

namespace printf_dbg {
namespace hana = boost::hana;
using namespace static_sized_str;
using namespace short_func;

template <typename T> struct FmtImpl;

template <typename T> struct IsHanaTupleImpl : std::false_type {};
template <typename... T>
struct IsHanaTupleImpl<hana::tuple<T...>> : std::true_type {};

template <typename T>
concept IsHanaTuple = IsHanaTupleImpl<T>::value;

template <typename T>
concept Formattable = requires(const T &v) {
  typename FmtImpl<std::decay_t<T>>;
  { FmtImpl<std::decay_t<T>>::fmt } -> IsStaticSizedStr;
  { FmtImpl<std::decay_t<T>>::vals(v) } -> IsHanaTuple;
};

template <Formattable T> constexpr auto fmt_t = FmtImpl<std::decay_t<T>>::fmt;

template <Formattable T> constexpr auto fmt_vals(const T &v) {
  return FmtImpl<std::decay_t<T>>::vals(v);
}

template <typename T> struct FmtImplSingle {
  static constexpr auto vals(T val) { return hana::make_tuple(val); }
};

template <std::unsigned_integral T> struct FmtImpl<T> : FmtImplSingle<T> {
  static constexpr auto fmt = [] {
    if constexpr (sizeof(T) <= sizeof(unsigned char)) {
      return s("%hhu");
    } else if constexpr (sizeof(T) <= sizeof(unsigned short)) {
      return s("%hu");
    } else if constexpr (sizeof(T) <= sizeof(unsigned)) {
      return s("%u");
    } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
      return s("%lu");
    } else {
      static_assert(sizeof(T) <= sizeof(unsigned long long),
                    "integral type is too large");
      return s("%llu");
    }
  }();
};

template <std::signed_integral T> struct FmtImpl<T> : FmtImplSingle<T> {
  static constexpr auto fmt = [] {
    if constexpr (sizeof(T) <= sizeof(char)) {
      return s("%hhi");
    } else if constexpr (sizeof(T) <= sizeof(short)) {
      return s("%hi");
    } else if constexpr (sizeof(T) <= sizeof(int)) {
      return s("%i");
    } else if constexpr (sizeof(T) <= sizeof(long)) {
      return s("%li");
    } else {
      static_assert(sizeof(T) <= sizeof(long long),
                    "integral type is too large");
      return s("%lli");
    }
  }();
};

// TODO: input somehow?
inline constexpr auto float_fmt = s("%g");

template <std::floating_point T> struct FmtImpl<T> : FmtImplSingle<T> {
  static constexpr auto fmt = float_fmt;
};

template <> struct FmtImpl<bool> {
  static constexpr auto fmt = s("%s");
  static constexpr auto vals(bool val) {
    return hana::make_tuple(val ? "true" : "false");
  }
};

template <> struct FmtImpl<const char *> : FmtImplSingle<const char *> {
  static constexpr auto fmt = s("%s");
};

template <> struct FmtImpl<char *> : FmtImpl<const char *> {};

template <typename T> struct FmtImpl<T *> {
  static constexpr auto fmt = s("%p");
  static constexpr auto vals(const T *val) {
    return hana::make_tuple(reinterpret_cast<const void *>(val));
  }
};

template <Formattable T, std::size_t size> struct FmtImpl<std::array<T, size>> {
  static constexpr auto fmt = s("{") + s(", ").join_n<size>(fmt_t<T>) + s("}");

  static constexpr auto vals(const std::array<T, size> &val) {
    return hana::unpack(
        val, [](const auto &...vals) { return hana::make_tuple(vals...); });
  }
};

// No good way to handle this for gpu...
inline constexpr bool is_colorized_output_enabled = false;

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
inline constexpr auto ANSI_EMPTY = s("");
inline constexpr auto ANSI_DEBUG = s("\x1b[02m");
// inline constexpr auto ANSI_WARN = s("\x1b[33m");
inline constexpr auto ANSI_EXPRESSION = s("\x1b[36m");
inline constexpr auto ANSI_VALUE = s("\x1b[01m");
inline constexpr auto ANSI_TYPE = s("\x1b[32m");
inline constexpr auto ANSI_RESET = s("\x1b[0m");

template <bool is_colorized, unsigned file_name_size, unsigned line_number_size,
          unsigned func_size, unsigned var_name_size>
constexpr auto
format_item_before(const StaticSizedStr<file_name_size> &file_name,
                   const StaticSizedStr<line_number_size> &line_number,
                   const StaticSizedStr<func_size> &func,
                   const StaticSizedStr<var_name_size> &var_name) {
  constexpr long max_file_name_len = 20;
  constexpr long to_remove =
      static_cast<long>(file_name_size) - max_file_name_len;
  auto file_name_reduced = [&] {
    if constexpr (to_remove > 0) {
      return s("..") + file_name.template remove_prefix<to_remove>();
    } else {
      return file_name;
    }
  }();

  auto ansi = [](auto code) {
    if constexpr (is_colorized) {
      return code;
    } else {
      return ANSI_EMPTY;
    }
  };

  return ansi(ANSI_RESET) + ansi(ANSI_DEBUG) + s("[") + file_name_reduced +
         s(":") + line_number + s(" (") + func + s(")] ") + ansi(ANSI_RESET) +
         ansi(ANSI_EXPRESSION) + var_name + ansi(ANSI_RESET) +
         ansi(ANSI_VALUE) + s(" =\n");
}

template <bool is_colorized, unsigned var_type_size>
constexpr auto
format_item_after(const StaticSizedStr<var_type_size> &var_type) {
  auto ansi = [](auto code) {
    if constexpr (is_colorized) {
      return code;
    } else {
      return ANSI_EMPTY;
    }
  };

  return ansi(ANSI_RESET) + ansi(ANSI_TYPE) + s("\n(") + var_type + s(")\n") +
         ansi(ANSI_RESET);
}

// here string_view must be null terminated (for usage with printf...)
template <bool is_colorized, typename... T, unsigned file_name_size,
          unsigned line_number_size, unsigned func_size,
          unsigned... var_name_sizes>
requires(sizeof...(T) == sizeof...(var_name_sizes)) constexpr auto debug_print_fmt(
    const StaticSizedStr<file_name_size> &file_name,
    const StaticSizedStr<line_number_size> &line_number,
    const StaticSizedStr<func_size> &func,
    const hana::tuple<StaticSizedStr<var_name_sizes>...> &var_names) {
  return hana::unpack(std::make_index_sequence<sizeof...(T)>{}, [&](auto... i) {
    return (... + [&](auto i) {
      return format_item_before<is_colorized>(file_name, line_number, func,
                                              var_names[i]) +
             fmt_t<PackElement<i, T...>> +
             format_item_after<is_colorized>(type_name<PackElement<i, T...>>);
    }(i));
  });
}

template <typename... T> constexpr decltype(auto) identity(T &&...vals) {
  return hana::back(hana::tuple<T &&...>{std::forward<T>(vals)...});
}

template <typename... T>
PRINTF_DBG_HOST_DEVICE decltype(auto) debug_print(const char *format_str,
                                                  T &&...vals) {
  hana::unpack(hana::flatten(hana::make_tuple(fmt_vals(vals)...)),
               [&](const auto &...vals) {
#ifdef __CUDA_ARCH__
                 static_assert(sizeof...(vals) <= 32,
                               "cuda printf accepts a maximum of 32 args!");
#endif
                 printf(format_str, vals...);
               });

  return identity(vals...);
}
} // namespace detail
} // namespace printf_dbg

#ifndef PRINTF_DBG_MACRO_DISABLE
#define PRINTF_DBG_TYPE(x) decltype(x)
#define PRINTF_DBG_TYPE_NAME(x) printf_dbg::type_name<decltype(x)>
#define PRINTF_DBG_STATIC_STR_CHAR_ARRAY(x)                                    \
  printf_dbg::StaticSizedStr<sizeof(x) - 1>(x)
#define PRINTF_DBG_STATIC_STRINGIFY(x)                                         \
  PRINTF_DBG_STATIC_STR_CHAR_ARRAY(MACRO_MAP_STRINGIFY(x))

// amusingly, forcing fmt to be constexpr dramatically reduces compile times
// when building for cuda because ptxas is very slow (at least for that
// sort of code) and constexpr allows for bypassing it totally...
#define PRINTF_DBG_IS_COLORIZED(is_colorized, ...)                             \
  do {                                                                         \
    constexpr auto fmt = printf_dbg::detail::debug_print_fmt<                  \
        is_colorized, MACRO_MAP_PP_COMMA_MAP(PRINTF_DBG_TYPE, __VA_ARGS__)>(   \
        PRINTF_DBG_STATIC_STR_CHAR_ARRAY(__FILE__),                            \
        PRINTF_DBG_STATIC_STRINGIFY(__LINE__),                                 \
        PRINTF_DBG_STATIC_STR_CHAR_ARRAY(__func__),                            \
        boost::hana::make_tuple(MACRO_MAP_PP_COMMA_MAP(                        \
            PRINTF_DBG_STATIC_STRINGIFY, __VA_ARGS__)));                       \
    printf_dbg::detail::debug_print(fmt.c_str(), __VA_ARGS__);                 \
  } while (0)

#define PRINTF_DBG_NO_COLOR(...) PRINTF_DBG_IS_COLORIZED(false, ##__VA_ARGS__)
#define PRINTF_DBG(...) PRINTF_DBG_IS_COLORIZED(true, ##__VA_ARGS__)
#else
#define PRINTF_DBG(...) printf_dbg::detail::identity(__VA_ARGS__)
#endif // PRINTF_DBG_MACRO_DISABLE
