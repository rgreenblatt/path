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

#include <concepts>
#include <tuple>
#include <utility>

template <typename T> struct AllValuesImpl;

template <typename T>
concept AllValuesEnumerable = requires {
  requires std::equality_comparable<T>;
  requires std::totally_ordered<T>;

  typename AllValuesImpl<T>;
  requires StdArrayOfType<decltype(AllValuesImpl<T>::values), T>;
};

template <typename T>
requires StdArrayOfType<decltype(T::all_values_array), T>
struct AllValuesImpl<T> {
  static constexpr auto values = T::all_values_array;
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

template <std::unsigned_integral T, T up_to> struct UpToGen {
  T value;
  constexpr UpToGen() : value{} {}
  constexpr UpToGen(T value) : value{value} { debug_assert(value < up_to); }
  constexpr operator T() const { return value; }
  constexpr operator T &() { return value; }
};

template <unsigned up_to> using UpTo = UpToGen<unsigned, up_to>;

template <std::unsigned_integral T, T up_to>
struct AllValuesImpl<UpToGen<T, up_to>> {
  static constexpr auto values = [] {
    std::array<UpToGen<T, up_to>, up_to> arr;
    for (std::size_t i = 0; i < arr.size(); ++i) {
      arr[i] = i;
    }

    return arr;
  }();
};

template <std::unsigned_integral T> struct AllValuesImpl<T> {
private:
  static constexpr T max = std::numeric_limits<T>::max();

public:
  static constexpr auto values = [] {
    std::array<T, static_cast<std::size_t>(max) + 1> arr;
    for (std::size_t i = 0; i < arr.size(); ++i) {
      arr[i] = i;
    }

    return arr;
  }();
};
