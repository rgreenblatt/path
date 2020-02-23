#pragma once

#include "lib/compile_time_dispatch/compile_time_dispatch.h"
#include "lib/compile_time_dispatch/to_array.h"

#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/for_each.hpp>

#include <tuple>

template <typename... Types>
struct CompileTimeDispatchable<std::tuple<Types...>> {
private:
  static constexpr unsigned num_elements = sizeof...(Types);

  using T = std::tuple<Types...>;

  template <unsigned i> using ElementT = std::tuple_element_t<i, T>;

public:
  static constexpr unsigned size =
      num_elements == 0
          ? 0
          : boost::hana::fold_left(
                std::make_index_sequence<num_elements>(), 1,
                [](unsigned size, auto idx) {
                  return size * CompileTimeDispatchable<ElementT<idx>>::size;
                });

  static constexpr std::array<T, size> values = [] {
    return to_array(boost::hana::cartesian_product(boost::hana::fold_left(
        std::make_index_sequence<num_elements>(), std::tuple<>{},
        [](auto tup, auto idx) {
          return std::tuple_cat(
              tup,
              std::make_tuple(CompileTimeDispatchable<ElementT<idx>>::values));
        })));
  }();
};
