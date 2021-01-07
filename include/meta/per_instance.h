#pragma once

#include "lib/assert.h"
#include "meta/all_values.h"

namespace per_instance {
namespace detail {
template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
struct PerInstanceImpl {
  // this approach is a bit gross, but I can't think of a better one...
  template <std::size_t... i>
  static constexpr auto items_helper(std::integer_sequence<std::size_t, i...>) {
    return V<TypeOver<AllValues<T>[i]>...>{};
  }

  using type = decltype(PerInstanceImpl::items_helper(
      std::make_index_sequence<AllValues<T>.size()>{}));
};
} // namespace detail
} // namespace per_instance

template <AllValuesEnumerable T, template <T> class TypeOver,
          template <typename...> class V>
using PerInstance =
    typename per_instance::detail::PerInstanceImpl<T, TypeOver, V>::type;
