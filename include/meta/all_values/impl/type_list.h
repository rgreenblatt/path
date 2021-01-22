#pragma once

#include "lib/array_transform.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/impl/range.h"
#include "meta/pack_element.h"
#include "meta/specialization_of.h"

template <typename... T> struct TypeList : UpTo<sizeof...(T)> {
  using UpTo<sizeof...(T)>::UpTo;
};

template <typename... T> struct AllValuesImpl<TypeList<T...>> {
  static constexpr auto values =
      convert_array<TypeList<T...>>(AllValues<UpTo<sizeof...(T)>>);
};

namespace type_list {
namespace detail {
template <typename T> struct TypeListTImpl;

template <typename... T> struct TypeListTImpl<TypeList<T...>> {
  template <TypeList<T...> idx> using Type = PackElement<idx, T...>;
};
} // namespace detail
} // namespace type_list

template <auto type>
requires SpecializationOf<decltype(type), TypeList>
using TypeListT = typename type_list::detail::TypeListTImpl<
    decltype(type)>::template Type<type>;
