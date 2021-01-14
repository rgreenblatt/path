#pragma once

#include "meta/all_types_same.h"
#include "meta/all_values.h"
#include "meta/tag.h"

namespace dispatch_name {
namespace detail {
template <typename T, typename F, unsigned... idxs>
concept TagDispatchablePack = AllValuesEnumerable<T> && requires(F &f) {
  requires AllTypesSame<decltype(f(Tag<T, idxs>{}))...>;
};

template <AllValuesEnumerable T, typename F, unsigned... idxs>
requires TagDispatchablePack<T, F, idxs...>
void check_tag_dispatchable(std::integer_sequence<unsigned, idxs...>);
} // namespace detail
} // namespace dispatch_name

template <typename T, typename F>
concept TagDispatchable = requires {
  requires AllValues<T>
  .size() != 0;
  dispatch_name::detail::check_tag_dispatchable<T, F>(
      std::make_integer_sequence<unsigned, AllValues<T>.size()>{});
};
