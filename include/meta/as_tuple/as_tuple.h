#pragma once

#include "meta/tuple.h"

#include <array>
#include <concepts>
#include <string_view>

template <typename T> struct AsTupleImpl;

template <typename T>
concept SpecializationAsTuple = requires(const T &t) {
  typename AsTupleImpl<T>;
  { AsTupleImpl<T>::as_tuple(t) } -> IsMetaTuple;
  requires requires(const decltype(AsTupleImpl<T>::as_tuple(t)) &tup) {
    { AsTupleImpl<T>::from_tuple(tup) } -> std::same_as<T>;
  };
};

template <typename T>
concept InternalAsTuple = requires(const T &t) {
  { t.as_tuple() } -> IsMetaTuple;
  requires requires(const decltype(t.as_tuple()) &tup) {
    { T::from_tuple(tup) } -> std::same_as<T>;
  };
};

// InternalAsTuple<T> here is just for clarity/better diagnostics
template <typename T>
concept AsTuple = InternalAsTuple<T> || SpecializationAsTuple<T>;

template <AsTuple T>
using AsTupleT = decltype(AsTupleImpl<T>::as_tuple(std::declval<T>()));

template <typename T>
concept SpecializationAsTupleStr = requires {
  requires AsTuple<T>;
  {
    AsTupleImpl<T>::as_tuple_strs()
    } -> std::same_as<
        std::array<std::string_view, meta_tuple_size_v<AsTupleT<T>>>>;
};

template <typename T>
concept InternalAsTupleStr = requires {
  requires AsTuple<T>;
  {
    T::as_tuple_strs()
    } -> std::same_as<
        std::array<std::string_view, meta_tuple_size_v<AsTupleT<T>>>>;
};

template <typename T>
concept AsTupleStr = InternalAsTupleStr<T> || SpecializationAsTupleStr<T>;

template <InternalAsTuple T> struct AsTupleImpl<T> {
  constexpr static auto as_tuple(const T &t) { return t.as_tuple(); }

  constexpr static T
  from_tuple(const decltype(std::declval<const T &>().as_tuple()) &tup) {
    return T::from_tuple(tup);
  }

  constexpr static auto as_tuple_strs() requires(InternalAsTupleStr<T>) {
    return T::as_tuple_strs();
  }
};
