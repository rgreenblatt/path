#pragma once

#include "lib/assert.h"
#include "meta/all_values.h"

namespace per_instance {
namespace detail {
template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
struct PerInstanceImpl {
  // this approach is a bit gross, but I can't think of a better one...
  template <std::size_t... i> struct ItemsHelper {
    using Type = V<TypeOver<AllValues<T>[i]>...>;
    constexpr ItemsHelper(std::integer_sequence<std::size_t, i...>) {}
  };

  template <std::size_t... i>
  ItemsHelper(std::integer_sequence<std::size_t, i...>) -> ItemsHelper<i...>;

  using type = typename decltype(ItemsHelper(
      std::make_index_sequence<AllValues<T>.size()>{}))::Type;
};
} // namespace detail
} // namespace per_instance

template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
using PerInstance =
    typename per_instance::detail::PerInstanceImpl<T, TypeOver, V>::type;
