#pragma once

#include "meta/all_values/all_values.h"
#include "meta/tuple.h"

#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/array.hpp>

template <> struct AllValuesImpl<MetaTuple<>> {
  static constexpr std::array<MetaTuple<>, 1> values = {{}};
};

template <AllValuesEnumerable... Types>
struct AllValuesImpl<MetaTuple<Types...>> {
  static constexpr auto values = boost::hana::unpack(
      boost::hana::cartesian_product(make_meta_tuple(AllValues<Types>...)),
      [](auto &&...x) { return std::array{x...}; });
};
