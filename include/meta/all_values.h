#pragma once

#include "meta/enum.h"
#include "meta/sequential_look_up.h"
#include "meta/std_array_specialization.h"
#include "meta/to_array.h"

#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <magic_enum.hpp>

#include <cassert>
#include <concepts>
#include <tuple>
#include <utility>

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

template <std::unsigned_integral T, T up_to> struct UpTo {
  T value;
  constexpr UpTo() : value{} { assert(value < up_to); }
  constexpr UpTo(T value) : value{value} { assert(value < up_to); }
  constexpr operator T() const { return value; }
  constexpr operator T &() { return value; }
};

template <std::unsigned_integral T, T up_to>
struct AllValuesImpl<UpTo<T, up_to>> {
  static constexpr auto values = boost::hana::unpack(
      std::make_integer_sequence<T, up_to>{}, [](auto... v) {
        return std::array<UpTo<T, up_to>, static_cast<std::size_t>(up_to) + 1>{
            v()...};
      });
};

template <std::unsigned_integral T> struct AllValuesImpl<T> {
private:
  static constexpr T max = std::numeric_limits<T>::max();

public:
  static constexpr auto values =
      boost::hana::unpack(std::make_integer_sequence<T, max>{}, [](auto... v) {
        return std::array<T, static_cast<std::size_t>(max) + 1>{
            static_cast<T>(v)..., max};
      });
};

template <AllValuesEnumerable E, E tag_in> struct Tag {
  static constexpr E tag = tag_in;
};
