#pragma once

#include "meta/all_values.h"
#include "meta/tag.h"

namespace dispatch_name {
namespace detail {
// Technically, these should check all the values not just the 0th....
// However, this will work unless the definition of F is very esoteric
template <AllValuesEnumerable T, typename F, unsigned idx, typename = void>
struct TTagDispatchableImpl : std::false_type {};


template <AllValuesEnumerable T, typename F, unsigned idx>
requires requires (F& f) {
  f(TTag<AllValues<T>[idx]>{});
}
struct TTagDispatchableImpl<T, F, idx> : std::true_type {};

template <AllValuesEnumerable T, typename F, unsigned... idxs>
requires(
    ... &&TTagDispatchableImpl<T, F, idxs>::
        value) void check_t_tag_dispatchable(std::integer_sequence<unsigned,
                                                                   idxs...>);

template <typename T, typename F, unsigned idx>
concept TagDispatchableAtIdx = AllValuesEnumerable<T> &&
    requires(F &f, Tag<T, idx> tag) {
  requires(AllValues<T>.size() != 0);
  f(tag);
};

template <AllValuesEnumerable T, typename F, unsigned... idxs>
requires(... &&TagDispatchableAtIdx<T, F, idxs>) void check_tag_dispatchable(
    std::integer_sequence<unsigned, idxs...>);
} // namespace detail
} // namespace dispatch_name

template <typename T, typename F>
concept TTagDispatchable = requires {
  AllValues<T>.size() != 0;
  dispatch_name::detail::check_t_tag_dispatchable<T, F>(
      std::make_integer_sequence<unsigned, AllValues<T>.size()>{});
};

template <typename T, typename F>
concept TagDispatchable = requires {
  AllValues<T>.size() != 0;
  dispatch_name::detail::check_tag_dispatchable<T, F>(
      std::make_integer_sequence<unsigned, AllValues<T>.size()>{});
};
