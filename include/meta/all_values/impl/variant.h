#pragma once

#include "lib/assert.h"
#include "meta/all_values/all_values.h"
#include "meta/array_cat.h"
#include "meta/pack_element.h"

#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/unpack.hpp>

#include <variant>

template <AllValuesEnumerable... Types>
struct AllValuesImpl<std::variant<Types...>> {
  static constexpr auto values = [] {
    constexpr unsigned num_elements = sizeof...(Types);
    using T = std::variant<Types...>;

    return boost::hana::unpack(
        std::make_index_sequence<num_elements>(), [](auto... idx) {
          auto out = array_cat([](auto idx) {
            return boost::hana::unpack(AllValues<PackElement<idx, Types...>>,
                                       [&](auto... v) {
                                         return std::array<T, sizeof...(v)>{
                                             T(std::in_place_index<idx>, v)...};
                                       });
          }(idx)...);

          static_assert(StdArraySpecialization<decltype(out)>);

          return out;
        });
  }();
};
