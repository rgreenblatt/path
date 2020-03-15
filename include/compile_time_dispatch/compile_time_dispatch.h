#pragma once

#include "lib/concepts.h"

#include "compile_time_dispatch/to_array.h"
#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/for_each.hpp>
#include <magic_enum.hpp>

#include <array>
#include <tuple>

template <typename T> struct CompileTimeDispatchableImpl;

template <typename T> concept CompileTimeDispatchable = requires {
  std::equality_comparable<T>;
  std::totally_ordered<T>;

  typename CompileTimeDispatchableImpl<T>;
  StdArraySpecialization<decltype(CompileTimeDispatchableImpl<T>::values)>;
  typename decltype(CompileTimeDispatchableImpl<T>::values)::value_type;
  std::same_as<
      typename decltype(CompileTimeDispatchableImpl<T>::values)::value_type, T>;
};

template <typename T> concept Enum = magic_enum::is_scoped_enum_v<T>;

template <Enum T> struct CompileTimeDispatchableImpl<T> {
  static constexpr auto values = magic_enum::enum_values<T>();
};

template <CompileTimeDispatchable... Types>
struct CompileTimeDispatchableImpl<std::tuple<Types...>> {
private:
  static constexpr unsigned num_elements = sizeof...(Types);

  using T = std::tuple<Types...>;

  template <unsigned i> using ElementT = std::tuple_element_t<i, T>;

public:
  static constexpr auto values = [] {
    return to_array(boost::hana::cartesian_product(boost::hana::fold_left(
        std::make_index_sequence<num_elements>(), std::tuple<>{},
        [](auto tup, auto idx) {
          return std::tuple_cat(
              tup, std::make_tuple(
                       CompileTimeDispatchableImpl<ElementT<idx>>::values));
        })));
  }();
};

template <CompileTimeDispatchable T>
struct CompileTimeDispatchableT : CompileTimeDispatchableImpl<T> {
  static constexpr std::size_t size =
      CompileTimeDispatchableImpl<T>::values.size();

  template <std::size_t idx>
  static constexpr T value = CompileTimeDispatchableImpl<T>::values[idx];

  static constexpr T index(std::size_t idx) {
    return CompileTimeDispatchableImpl<T>::values[idx];
  }
};
