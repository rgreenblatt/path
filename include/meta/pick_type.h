#pragma once

#include "meta/all_values.h"
#include "meta/get_idx.h"
#include "meta/pack_element.h"

template <auto type, typename... T>
requires(AllValuesEnumerable<decltype(type)> &&
         sizeof...(T) == AllValues<decltype(type)>.size()) using PickType =
    PackElement<get_idx(type), T...>;
