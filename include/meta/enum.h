#pragma once

#include <magic_enum.hpp>

template <typename T>
concept Enum = magic_enum::is_scoped_enum_v<T>;
