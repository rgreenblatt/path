#pragma once

#include "meta/as_tuple.h"
#include "meta/tuple.h"

#include <boost/hana/unpack.hpp>

// TODO: constexpr???
#define AS_TUPLE_STRUCTURAL(NAME, ...)                                         \
  constexpr auto as_tuple() const { return make_meta_tuple(__VA_ARGS__); }     \
                                                                               \
  constexpr static NAME from_tuple(                                            \
      const decltype(make_meta_tuple(__VA_ARGS__)) &tup) {                     \
    return boost::hana::unpack(                                                \
        tup, [](const auto &...vals) -> NAME { return {vals...}; });           \
  }
