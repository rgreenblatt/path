#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "meta/aggregate_constructible_from.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/per_instance.h"
#include "meta/all_values/sequential_dispatch.h"
#include "meta/all_values/tag.h"
#include "meta/decays_to.h"
#include "meta/pack_element.h"
#include "meta/specialization_of.h"
#include "meta/std_array_specialization.h"

#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/unpack.hpp>

#include <concepts>
#include <type_traits>
#include <utility>
#include <variant>

template <AllValuesEnumerable E, typename... T>
requires(AllValues<E>.size() == sizeof...(T) &&
         sizeof...(T) > 0) struct TaggedUnion {
  static constexpr auto values = AllValues<E>;

  template <unsigned idx> using Type = PackElement<idx, T...>;

  std::variant<T...> item;

  constexpr TaggedUnion() = default;

  constexpr explicit TaggedUnion(std::variant<T...> item)
      : item{std::move(item)} {}

  template <unsigned idx, typename... Args>
  requires(AggregateConstructibleFrom<Type<idx>, Args...>
               &&std::movable<Type<idx>>) constexpr TaggedUnion(Tag<E, idx>,
                                                                Args &&...args)
      : item{std::in_place_index_t<idx>{},
             std::move(Type<idx>{std::forward<Args>(args)...})} {}

  template <unsigned idx>
  requires std::movable<Type<idx>>
  constexpr TaggedUnion(Tag<E, idx>, Type<idx> v)
      : item{std::in_place_index_t<idx>{}, std::move(v)} {}

  ATTR_PURE_NDEBUG constexpr unsigned idx() const { return item.index(); }

  ATTR_PURE_NDEBUG constexpr E type() const { return values[idx()]; }

  // we have to roll our own visit because std::visit isn't constexpr
  template <typename U, typename F>
  constexpr static decltype(auto) static_visit(U &&v, F &&f) {
    return sequential_dispatch<values.size()>(v.item.index(), [&](auto i) {
      constexpr Tag<E, i> tag;
      return f(tag, v.get(tag));
    });
  }

  template <unsigned idx, typename U>
  ATTR_PURE_NDEBUG constexpr static decltype(auto) static_get(U &&v) {
    debug_assert_assume(idx == v.idx());
    return *std::get_if<idx>(&v.item);
  }

  template <typename F> constexpr decltype(auto) visit_tagged(F &&f) {
    return static_visit(*this, std::forward<F>(f));
  }

  template <typename F> constexpr decltype(auto) visit_tagged(F &&f) const {
    return static_visit(*this, std::forward<F>(f));
  }

  template <typename F> constexpr decltype(auto) visit(F &&f) const {
    return visit_tagged([&](auto, auto &&v) { return f(v); });
  }

  template <typename F> constexpr decltype(auto) visit(F &&f) {
    return visit_tagged([&](auto, auto &&v) { return f(v); });
  }

  template <typename F> constexpr decltype(auto) visit_indexed(F &&f) {
    return visit([&](auto &&v) { return f(type(), v); });
  }

  template <typename F> constexpr decltype(auto) visit_indexed(F &&f) const {
    return visit([&](auto &&v) { return f(type(), v); });
  }

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) {
    return static_get<idx>(*this);
  }

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) const {
    return static_get<idx>(*this);
  }

  constexpr bool operator==(const TaggedUnion &other) const = default;
  constexpr auto operator<=>(const TaggedUnion &other) const = default;
  // TODO: hana tuple compare work around...
  constexpr bool operator<(const TaggedUnion &other) const = default;
};

template <AllValuesEnumerable T, template <T> class TypeOver>
using TaggedUnionPerInstance = PerInstanceTakesType<T, TypeOver, TaggedUnion>;

template <AllValuesEnumerable E, AllValuesEnumerable... Types>
struct AllValuesImpl<TaggedUnion<E, Types...>> {
private:
  static constexpr unsigned num_elements = sizeof...(Types);

  using T = TaggedUnion<E, Types...>;

  static constexpr auto tag_values = AllValues<E>;

  template <typename T, std::size_t... sizes>
  static constexpr auto array_cat(const std::array<T, sizes> &...arr) {
    constexpr std::size_t out_size = (... + sizes);
    std::array<T, out_size> out;
    unsigned start = 0;
    ((std::copy(arr.begin(), arr.end(), out.begin() + start),
      start += arr.size()),
     ...);
    debug_assert_assume(start == out_size);

    return out;
  }

public:
  static constexpr auto values = [] {
    return boost::hana::unpack(
        std::make_index_sequence<num_elements>(), [](auto... idx) {
          auto out = array_cat([](auto idx) {
            return boost::hana::unpack(AllValues<PackElement<idx, Types...>>,
                                       [&](auto... v) {
                                         return std::array<T, sizeof...(v)>{
                                             T(tag_v<tag_values[idx]>, v)...};
                                       });
          }(idx)...);

          static_assert(StdArraySpecialization<decltype(out)>);

          return out;
        });
  }();
};
