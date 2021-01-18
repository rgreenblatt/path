#pragma once

#include "meta/pack_element.h"
#include <concepts>

template <typename... Rest>
concept AllTypesSame = sizeof...(Rest) == 0 ||
                       (... && std::same_as<PackElement<0, Rest...>, Rest>);
