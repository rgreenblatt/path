#pragma once

#include "meta/all_values.h"
#include "meta/to_array.h"
#include "meta/tuple.h"

#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/array.hpp>

template <> struct AllValuesImpl<MetaTuple<>> {
  static constexpr std::array<MetaTuple<>, 1> values = {{}};
};

template <AllValuesEnumerable... Types>
struct AllValuesImpl<MetaTuple<Types...>> {
  static constexpr auto values = to_array(
      boost::hana::cartesian_product(make_meta_tuple(AllValues<Types>...)));
};
