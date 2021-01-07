#pragma once

#ifndef __clang__
#include <tuple>
#endif

template <unsigned idx, typename... T>
using PackElement =
#ifdef __clang__
    __type_pack_element<idx, T...>;
#else
    std::tuple_element_t<idx, std::tuple<T...>>;
#endif
