#pragma once

#include "meta/all_types_same.h"
#include "meta/all_values.h"
#include "meta/tag.h"

#include <utility>

namespace dispatch_name {
namespace detail {
template <typename F, typename T, unsigned... idxs>
concept TagDispatchablePack = AllValuesEnumerable<T> && requires(F &f) {
  requires AllTypesSame<decltype(f(Tag<T, idxs>{}))...>;
};

template <typename F, typename T, typename Idxs> struct CheckTagDispatchable;

template <typename F, AllValuesEnumerable T, std::size_t... idxs>
struct CheckTagDispatchable<F, T, std::index_sequence<idxs...>> {
  static constexpr bool value = TagDispatchablePack<F, T, idxs...>;
};
} // namespace detail
} // namespace dispatch_name

template <typename F, typename T>
concept TagDispatchable = requires {
  requires AllValues<T>
  .size() != 0;
  requires dispatch_name::detail::CheckTagDispatchable < F, T,
      std::make_index_sequence < AllValues<T>
  .size() >> ::value;
};
