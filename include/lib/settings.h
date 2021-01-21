#pragma once

#include "meta/as_tuple/str_macro.h"

#include <concepts>

template <typename T>
concept Setting = std::regular<T> && AsTupleStr<T>;

// TODO: constexpr ???
#define SETTING_BODY(NAME, ...)                                                \
  AS_TUPLE_STR_STRUCTURAL(NAME, ##__VA_ARGS__)                                 \
  constexpr bool operator==(const NAME &) const = default;

struct EmptySettings {
  SETTING_BODY(EmptySettings);
};

static_assert(Setting<EmptySettings>);
