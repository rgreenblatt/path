#pragma once

#include "meta/all_values.h"
#include "meta/tuple.h"
#include "meta/to_array.h"

#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/cartesian_product.hpp>

template <AllValuesEnumerable... Types>
struct AllValuesImpl<MetaTuple<Types...>> {
  static constexpr auto values = to_array(
      boost::hana::cartesian_product(make_meta_tuple(AllValues<Types>...)));
};
