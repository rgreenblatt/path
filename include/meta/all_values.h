#pragma once

#include "lib/tagged_union.h"
#include "meta/concepts.h"
#include "meta/enum.h"
#include "meta/to_array.h"

#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/unpack.hpp>
#include <magic_enum.hpp>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace detail {
template <typename T> struct AllValuesImpl;
}

template <typename T> concept AllValuesEnumerable = requires {
  requires std::equality_comparable<T>;
  requires std::totally_ordered<T>;

  typename detail::AllValuesImpl<T>;
  requires StdArrayOfType<decltype(detail::AllValuesImpl<T>::values), T>;
};

template <AllValuesEnumerable T>
constexpr auto AllValues = detail::AllValuesImpl<T>::values;

// implementations...
namespace detail {
template <Enum T> struct AllValuesImpl<T> {
  static constexpr auto values = magic_enum::enum_values<T>();
};

template <AllValuesEnumerable... Types>
struct AllValuesImpl<std::tuple<Types...>> {
private:
  static constexpr unsigned num_elements = sizeof...(Types);

  using T = std::tuple<Types...>;

  template <unsigned i> using ElementT = __type_pack_element<i, T>;

public:
  static constexpr auto values = to_array(
      boost::hana::cartesian_product(std::make_tuple(AllValues<Types>...)));
};

template <Enum E, AllValuesEnumerable... Types>
struct AllValuesImpl<TaggedUnion<E, Types...>> {
private:
  static constexpr unsigned num_elements = sizeof...(Types);

  using T = TaggedUnion<E, Types...>;

public:
  static constexpr auto values = [] {
    return boost::hana::fold_left(
        std::make_index_sequence<num_elements>(), std::array<T, 0>{},
        [](auto arr, auto idx) {
          const auto &values = AllValues<__type_pack_element<idx, T>>;
          std::array<T, values.size()> out_values;
          for (int i = 0; i < values.size(); ++i) {
            out_values[i] = T::T<magic_enum::enum_value<E>(idx)>(values[i]);
          }

          return std::tuple_cat(arr, out_values);
        });
  }();
};
} // namespace detail
