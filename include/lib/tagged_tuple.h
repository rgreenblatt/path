#pragma once

#include "meta/all_values.h"
#include "meta/decays_to.h"
#include "meta/get_idx.h"
#include "meta/per_instance.h"
#include "meta/sequential_dispatch.h"
#include "meta/tag.h"

#include <compare>
#include <tuple>
#include <utility>

template <AllValuesEnumerable E, typename... T> struct TaggedTuple {
  std::tuple<T...> items;

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) {
    return std::get<idx>(items);
  }

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) const {
    return std::get<idx>(items);
  }

  template <typename F> constexpr auto visit(const F &f, const E &value) {
    return visit_impl(f, value, *this);
  }

  template <typename F> constexpr auto visit(const F &f, const E &value) const {
    return visit_impl(f, value, *this);
  }

  constexpr bool operator==(const TaggedTuple &other) const = default;
  constexpr auto operator<=>(const TaggedTuple &other) const = default;

private:
  template <typename F, DecaysTo<TaggedTuple> V>
  static constexpr auto visit_impl(const F &f, const E &type, V &&v) {
    return sequential_dispatch<AllValues<E>.size()>(
        get_idx(type),
        [&]<unsigned idx>(NTag<idx>) { return f(std::get<idx>(v.items)); });
  }
};

template <AllValuesEnumerable T, template <T> class TypeOver>
using TaggedTuplePerInstance = PerInstanceTakesType<T, TypeOver, TaggedTuple>;
