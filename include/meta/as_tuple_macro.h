#pragma once

#include "meta/as_tuple.h"
#include "meta/tuple.h"
#include "meta/unpack_to.h"

// TODO: constexpr???
#define AS_TUPLE_STRUCTURAL(NAME, ...)                                         \
  constexpr auto as_tuple() const { return make_meta_tuple(__VA_ARGS__); }     \
                                                                               \
  constexpr static NAME from_tuple(                                            \
      const decltype(make_meta_tuple(__VA_ARGS__)) &tup) {                     \
    return unpack_to<NAME>(tup);                                               \
  }
