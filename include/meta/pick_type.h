#include "lib/enum.h"
#include <magic_enum.hpp>

template <Enum E, E type, typename... T>
requires (sizeof...(T) == magic_enum::enum_count<E>())
using PickType = __type_pack_element<magic_enum::enum_integer(type), T...>;
