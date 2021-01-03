#pragma once

#include "all_values.h"

namespace per_instance {
namespace detail {
template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
struct PerInstanceImpl {
  static constexpr auto values = AllValues<T>;
  static constexpr unsigned size = AllValues<T>.size();

  // TODO: this approach is gross and would fail if the type isn't default
  // initializable...
  template <std::size_t... i>
  static constexpr auto items_helper(std::integer_sequence<std::size_t, i...>) {
    return V<TypeOver<values[i]>...>{};
  }

  using type =
      decltype(PerInstanceImpl::items_helper(std::make_index_sequence<size>{}));
};
} // namespace detail
} // namespace per_instance

template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
using PerInstance =
    typename per_instance::detail::PerInstanceImpl<T, TypeOver, V>::type;
