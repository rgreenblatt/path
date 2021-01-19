#pragma once

#include "meta/all_values.h"
#include "meta/all_values_tuple.h"
#include "meta/as_tuple.h"
#include "meta/tuple.h"

template <AsTuple T> struct AllValuesImpl<T> {
  static constexpr auto values = [] {
    constexpr auto tuple_values = AllValues<AsTupleT<T>>;
    std::array<T, tuple_values.size()> out;
    std::transform(tuple_values.begin(), tuple_values.end(), out.begin(),
                   [](auto in) { return AsTupleImpl<T>::from_tuple(in); });

    return out;
  }();
};
