#pragma once

#include "meta/all_values.h"
#include "meta/decays_to.h"
#include "meta/get_idx.h"
#include "meta/per_instance.h"
#include "meta/sequential_dispatch.h"
#include "meta/tag.h"
#include "meta/tuple.h"
#include "meta/all_values_tuple.h"

#include <boost/hana/unpack.hpp>
#include <boost/hana/ext/std/array.hpp>

#include <compare>
#include <array>

template <AllValuesEnumerable E, typename... T> struct TaggedTuple {
  MetaTuple<T...> items;

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) {
    return items[boost::hana::size_c<idx>];
  }

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) const {
    return items[boost::hana::size_c<idx>];
  }

  template <typename F> constexpr auto visit(F &&f, const E &value) {
    return visit_impl(std::forward<F>(f), value, *this);
  }

  template <typename F> constexpr auto visit(F &&f, const E &value) const {
    return visit_impl(std::forward<F>(f), value, *this);
  }

  constexpr bool operator==(const TaggedTuple &other) const = default;
  constexpr auto operator<=>(const TaggedTuple &other) const = default;

private:
  template <typename F, DecaysTo<TaggedTuple> V>
  static constexpr auto visit_impl(F &&f, const E &type, V &&v) {
    return sequential_dispatch<AllValues<E>.size()>(
        get_idx(type), [&](auto tag) { return f(v.get(to_tag<E>(tag))); });
  }
};

template <AllValuesEnumerable T, template <T> class TypeOver>
using TaggedTuplePerInstance = PerInstanceTakesType<T, TypeOver, TaggedTuple>;

template<AllValuesEnumerable E, typename... T>
struct AllValuesImpl<TaggedTuple<E, T...>> {
  static constexpr auto values =
      boost::hana::unpack(AllValues<MetaTuple<T...>>, [](auto... values) {
        return std::array{TaggedTuple<E, T...>{values}...};
      });
};
