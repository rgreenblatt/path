#pragma once

#include "meta/concepts.h"
#include "meta/enum.h"
#include "meta/to_array.h"

#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <magic_enum.hpp>

#include <tuple>
#include <concepts>

template <typename T> struct AllValuesImpl;

template <typename T> concept AllValuesEnumerable = requires {
  requires std::equality_comparable<T>;
  requires std::totally_ordered<T>;

  typename AllValuesImpl<T>;
  requires StdArrayOfType<decltype(AllValuesImpl<T>::values), T>;
};

template <AllValuesEnumerable T>
constexpr auto AllValues = AllValuesImpl<T>::values;

// implementations...
template <Enum T> struct AllValuesImpl<T> {
  static constexpr auto values = magic_enum::enum_values<T>();
};

template <AllValuesEnumerable... Types>
struct AllValuesImpl<std::tuple<Types...>> {
  static constexpr auto values = to_array(
      boost::hana::cartesian_product(std::make_tuple(AllValues<Types>...)));
};
