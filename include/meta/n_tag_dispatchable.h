#pragma once

#include "meta/all_types_same.h"
#include "meta/n_tag.h"

#include <utility>

namespace dispatch_name {
namespace detail {
template <typename F, unsigned... idxs>
concept NTagDispatchablePack = requires(F &&f) {
  // TODO: gcc work around (should work on trunk)
#ifdef __clang__
  requires AllTypesSame<decltype(f(NTag<idxs>{}))...>;
#else
  requires true;
#endif
};

template <typename F, typename Idxs> struct CheckNTagDispatchable;

template <typename F, std::size_t... idxs>
struct CheckNTagDispatchable<F, std::index_sequence<idxs...>> {
  static constexpr bool value = NTagDispatchablePack<F, idxs...>;
};
} // namespace detail
} // namespace dispatch_name

template <typename F, unsigned size>
concept NTagDispatchable = requires {
  requires size != 0;
  requires dispatch_name::detail::CheckNTagDispatchable<
      F, std::make_index_sequence<size>>::value;
};
