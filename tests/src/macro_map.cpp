#include "meta/macro_map.h"
#include "lib/info/printf_dbg.h"
#include "meta/tuple.h"

#include <gtest/gtest.h>

#include <string_view>

using namespace std::literals;

#define STRING_VIEWIFY(x) std::string_view(MACRO_MAP_STRINGIFY(x))
#define STRING_VIEW_TUP(...)                                                   \
  boost::hana::make_tuple(MACRO_MAP_PP_COMMA_MAP(STRING_VIEWIFY, __VA_ARGS__))

#define ONE_EXPAND(x) 1
#define PLUS_EXPAND() +
#define COUNT(...) MACRO_MAP_PP_MAP(ONE_EXPAND, PLUS_EXPAND, __VA_ARGS__)

static_assert(COUNT(a, b, asd) == 3);
static_assert(STRING_VIEW_TUP() == make_meta_tuple());
static_assert(STRING_VIEW_TUP(a, b, asd) ==
              make_meta_tuple("a"sv, "b"sv, "asd"sv));
static_assert(MACRO_MAP_PP_NARG() == 0);
static_assert(MACRO_MAP_PP_NARG(a) == 1);
static_assert(MACRO_MAP_PP_NARG(a, b, c) == 3);
static_assert(MACRO_MAP_PP_NARG(a, a, a, a, a, a) == 6);

// should print - not a real test
TEST(MACRO_MAP, printf_dbg) {
  int a = 12;
  PRINTF_DBG(a, a + 8, 9 + 10);
}
