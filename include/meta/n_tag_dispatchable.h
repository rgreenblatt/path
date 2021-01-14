#pragma once

#include "meta/all_types_same.h"
#include "meta/n_tag.h"

#include <utility>

namespace dispatch_name {
namespace detail {
template <typename F, unsigned... idxs>
concept NTagDispatchablePack = requires(F &&f) {
  requires AllTypesSame<decltype(f(NTag<idxs>{}))...>;
};

template <typename F, unsigned... idxs>
requires NTagDispatchablePack<F, idxs...>
void check_n_tag_dispatchable(std::integer_sequence<unsigned, idxs...>);
} // namespace detail
} // namespace dispatch_name

template <unsigned size, typename F>
concept NTagDispatchable = requires {
  requires size != 0;
  dispatch_name::detail::check_n_tag_dispatchable<F>(
      std::make_integer_sequence<unsigned, size>{});
};
