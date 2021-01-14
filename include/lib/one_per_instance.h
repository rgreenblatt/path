#pragma once

#include "meta/all_values.h"
#include "meta/get_idx.h"
#include "meta/per_instance.h"
#include "meta/sequential_dispatch.h"
#include "meta/tag.h"

#include <compare>
#include <tuple>
#include <utility>

template <AllValuesEnumerable E, template <E> class TypeOver>
struct OnePerInstance {
  using Items = PerInstance<E, TypeOver, std::tuple>;

  Items items;

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) {
    return std::get<idx>(items);
  }

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) const {
    return std::get<idx>(items);
  }

  template <typename F> constexpr auto visit(const F &f, const E &value) {
    return sequential_dispatch<AllValues<E>.size()>(
        get_idx(value),
        [&]<unsigned idx>(NTag<idx>) { return f(std::get<idx>(items)); });
  }

  constexpr bool operator==(const OnePerInstance &other) const = default;
  constexpr auto operator<=>(const OnePerInstance &other) const = default;
};
