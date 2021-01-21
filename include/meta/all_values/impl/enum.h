#pragma once

#include "meta/all_values/all_values.h"
#include "meta/enum.h"

#include <magic_enum.hpp>

// implementations...
template <Enum T> struct AllValuesImpl<T> {
  static constexpr auto values = magic_enum::enum_values<T>();
};
