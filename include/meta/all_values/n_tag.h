#pragma once

#include <type_traits>

template <unsigned idx> using NTag = std::integral_constant<unsigned, idx>;
