#pragma once

#include "lib/assert.h"
#include "meta/all_values/all_values.h"

namespace per_instance {
namespace detail {
template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
struct PerInstanceImpl {
  template <typename> struct ItemsHelper;

  template <std::size_t... i> struct ItemsHelper<std::index_sequence<i...>> {
    using Type = V<TypeOver<AllValues<T>[i]>...>;
  };

  using type =
      typename ItemsHelper<std::make_index_sequence<AllValues<T>.size()>>::Type;
};

template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class VTakesType>
struct PerInstanceTakesTypeImpl {
  template <typename... Ts> using V = VTakesType<T, Ts...>;
  using type = typename PerInstanceImpl<T, TypeOver, V>::type;
};
} // namespace detail
} // namespace per_instance

template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
using PerInstance =
    typename per_instance::detail::PerInstanceImpl<T, TypeOver, V>::type;

template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
using PerInstanceTakesType =
    typename per_instance::detail::PerInstanceTakesTypeImpl<T, TypeOver,
                                                            V>::type;
