#pragma once

#include "compile_time_dispatch/compile_time_dispatch.h"

#include <magic_enum.hpp>

template <typename Enum>
struct CompileTimeDispatchable<
    Enum, std::enable_if_t<magic_enum::is_scoped_enum_v<Enum>>> {
  static constexpr unsigned size = magic_enum::enum_count<Enum>();

  static constexpr auto values = magic_enum::enum_values<Enum>();
};
