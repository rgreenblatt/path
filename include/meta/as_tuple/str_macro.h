#pragma once

#include "meta/as_tuple/macro.h"
#include "meta/macro_map.h"

#define AS_TUPLE_STR_STRUCTURAL(NAME, ...)                                     \
  AS_TUPLE_STRUCTURAL(NAME, ##__VA_ARGS__)                                     \
  constexpr static std::string_view type_name() { return #NAME; }              \
  constexpr static std::array<std::string_view,                                \
                              MACRO_MAP_PP_NARG(__VA_ARGS__)>                  \
  as_tuple_strs() {                                                            \
    return {MACRO_MAP_STRINGIFY_COMMA(__VA_ARGS__)};                           \
  }
