#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "meta/aggregate_constructible_from.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/get_idx.h"
#include "meta/all_values/per_instance.h"
#include "meta/all_values/sequential_dispatch.h"
#include "meta/all_values/tag.h"
#include "meta/decays_to.h"
#include "meta/pack_element.h"
#include "meta/specialization_of.h"
#include "meta/std_array_specialization.h"

#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/unpack.hpp>

#include <concepts>
#include <type_traits>

// This really should just use std::variant internally (which would make this
// muchhhhh less painful), but due to clang bugs and bugs in libstdc++, that
// isn't possible with constexpr
//
// Also, there isn't a version of std::visit(std::variant) which is constexpr
// (but that can be worked around...)

namespace tagged_union {
namespace detail {
template <typename... Rest> union VariadicUnion;

template <> union VariadicUnion<> {};

template <typename First, typename... Rest>
union VariadicUnion<First, Rest...> {
private:
  static constexpr bool trivially_destructible =
      std::is_trivially_destructible_v<First> &&
      (... && std::is_trivially_destructible_v<Rest>);
  static constexpr bool destructible =
      std::is_destructible_v<First> && (... && std::is_destructible_v<Rest>);

public:
  constexpr VariadicUnion() : first_{} {}

  constexpr ~VariadicUnion() requires trivially_destructible = default;

  // delegated to holder - must be manually destructed to avoid UB
  constexpr ~VariadicUnion() requires(!trivially_destructible && destructible) {
  }

  template <std::size_t idx, typename... Args>
  requires(idx <=
           sizeof...(Rest)) constexpr VariadicUnion(std::in_place_index_t<idx>,
                                                    Args &&...args)
      : rest_(std::in_place_index_t<idx - 1u>{}, std::forward<Args>(args)...) {}

  template <typename... Args>
  constexpr VariadicUnion(std::in_place_index_t<0u>, Args &&...args)
      : first_{std::forward<Args>(args)...} {}

private:
  First first_;
  VariadicUnion<Rest...> rest_;

  friend struct access;
};

struct access {
  template <unsigned idx, SpecializationOf<VariadicUnion> UnionType>
  static constexpr auto &get(UnionType &&v) {
    if constexpr (idx == 0) {
      return v.first_;
    } else {
      return get<idx - 1>(v.rest_);
    }
  }
};
} // namespace detail
} // namespace tagged_union

template <AllValuesEnumerable E, typename... T>
requires(AllValues<E>.size() == sizeof...(T) &&
         sizeof...(T) > 0) class TaggedUnion {
private:
  static constexpr auto values = AllValues<E>;

  using ac = tagged_union::detail::access;

  template <unsigned idx> using Type = PackElement<idx, T...>;

  template <DecaysTo<TaggedUnion> First, DecaysTo<TaggedUnion>... Rest,
            typename F>
  static constexpr decltype(auto) visit_n(F &&f, First &&first,
                                          Rest &&...rest) {
    debug_assert(((first.idx_ == rest.idx_) && ... && true));
    return [&](auto &&...vals) {
      return sequential_dispatch<values.size()>(
          first.idx_, [&]<unsigned idx>(NTag<idx>) {
            return f(Tag<E, idx>{}, ac::get<idx>(vals.union_)...);
          });
    }(first, rest...);
  }

  static constexpr bool trivially_destructible =
      (... && std::is_trivially_destructible_v<T>);

  static constexpr bool destructible = (... && std::is_destructible_v<T>);

  static constexpr bool trivially_move_constructible =
      (... && std::is_trivially_move_constructible_v<T>);

  static constexpr bool move_constructible =
      (... && std::is_move_constructible_v<T>);

  static constexpr bool trivially_copy_constructible =
      (... && std::is_trivially_copy_constructible_v<T>);

  static constexpr bool copy_constructible =
      (... && std::is_copy_constructible_v<T>);

  static constexpr bool trivially_move_assignable =
      (... && std::is_trivially_move_assignable_v<T>);

  static constexpr bool move_assignable = (... && std::is_move_assignable_v<T>);

  static constexpr bool trivially_copy_assignable =
      (... && std::is_trivially_copy_assignable_v<T>);

  static constexpr bool copy_assignable = (... && std::is_copy_assignable_v<T>);

public:
  using TagType = E;

  template <unsigned idx, typename... Args>
  requires(AggregateConstructibleFrom<
           Type<idx>, Args...>) constexpr TaggedUnion(Tag<E, idx>,
                                                      Args &&...args)
      : idx_(idx),
        union_(std::in_place_index_t<idx>{}, std::forward<Args>(args)...) {}

  template <unsigned idx>
  requires std::copyable<Type<idx>>
  constexpr TaggedUnion(Tag<E, idx>, const Type<idx> &v)
      : idx_(idx), union_(std::in_place_index_t<idx>{}, v) {}

  template <unsigned idx>
  requires std::movable<Type<idx>>
  constexpr TaggedUnion(Tag<E, idx>, Type<idx> &&v)
      : idx_(idx),
        union_(std::in_place_index_t<idx>{}, std::forward<Type<idx>>(v)) {}

  constexpr TaggedUnion() : TaggedUnion(Tag<E, 0>{}) {}

  constexpr TaggedUnion(const TaggedUnion &other) requires
      trivially_copy_constructible = default;

  constexpr TaggedUnion(const TaggedUnion &other) requires(
      !trivially_copy_constructible && copy_constructible)
      : idx_(other.idx_) {
    visit_n(
        [](auto holder, auto &&l, auto &&r) {
          new (&l) PackElement<decltype(holder)::idx, T...>(r);
        },
        *this, other);
  }

  constexpr TaggedUnion(TaggedUnion &&other) requires(
      trivially_move_constructible) = default;

  constexpr TaggedUnion(TaggedUnion &&other) requires(
      !trivially_move_constructible && move_constructible)
      : idx_(other.idx_) {
    visit_n(
        [](auto holder, auto &&l, auto &&r) {
          new (&l) PackElement<decltype(holder)::idx, T...>(std::move(r));
        },
        *this, std::forward<TaggedUnion>(other));
  }

  constexpr TaggedUnion &operator=(const TaggedUnion &other) requires
      trivially_copy_assignable = default;

  constexpr TaggedUnion &
  operator=(const TaggedUnion &other) requires(!trivially_copy_assignable &&
                                               copy_assignable) {
    if (this != &other) {
      // this is plausibly suboptimal if idx_ == other.idx_, should
      // theoretically use the type's operator= to avoid having to call
      // destructor
      this->TaggedUnion::~TaggedUnion();
      idx_ = other.idx_;
      visit_n(
          [](auto holder, auto &&l, auto &&r) {
            new (&l) PackElement<decltype(holder)::idx, T...>(r);
          },
          *this, other);
    }
    return *this;
  }

  constexpr TaggedUnion &
  operator=(TaggedUnion &&other) requires(trivially_move_assignable) = default;

  constexpr TaggedUnion &
  operator=(TaggedUnion &&other) requires(!trivially_move_assignable &&
                                          move_assignable) {
    if (this != &other) {
      // this is plausibly suboptimal if idx_ == other.idx_, should
      // theoretically use the type's operator= to avoid having to call
      // destructor
      this->TaggedUnion::~TaggedUnion();
      idx_ = other.idx_;
      visit_n(
          [](auto holder, auto &&l, auto &&r) {
            new (&l) PackElement<decltype(holder)::idx, T...>(std::move(r));
          },
          *this, std::forward<TaggedUnion>(other));
    }
    return *this;
  }

  constexpr ~TaggedUnion() requires(trivially_destructible) = default;

  constexpr ~TaggedUnion() requires(!trivially_destructible && destructible) {
    visit([](auto &in) { destroy_input(in); });
  }

  ATTR_PURE_NDEBUG constexpr E type() const { return values[idx_]; }

  template <typename F> constexpr decltype(auto) visit_tagged(F &&f) {
    return visit_n(std::forward<F>(f), *this);
  }

  template <typename F> constexpr decltype(auto) visit_tagged(F &&f) const {
    return visit_n(std::forward<F>(f), *this);
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
    debug_assert_assume(idx == idx_);
    return ac::get<idx>(union_);
  }

  template <unsigned idx>
  ATTR_PURE_NDEBUG constexpr decltype(auto) get(Tag<E, idx>) const {
    debug_assert_assume(idx == idx_);
    return ac::get<idx>(union_);
  }

  constexpr auto operator<=>(const TaggedUnion &other) const
      requires(... &&std::totally_ordered<T>) {
    if (idx_ != other.idx_) {
      return idx_ <=> other.idx_;
    }

    return visit_n([](auto, const auto &l, const auto &r) { return l <=> r; },
                   *this, other);
  }

  constexpr bool operator==(const TaggedUnion &other) const
      requires(... &&std::equality_comparable<T>) {
    if (idx_ != other.idx_) {
      return false;
    }

    return visit_n([](auto, const auto &l, const auto &r) { return l == r; },
                   *this, other);
  }

  // TODO: hana tuple compare work around...
  constexpr auto operator<(const TaggedUnion &other) const
      requires(... &&LessComparable<T>) {
    if (idx_ != other.idx_) {
      return idx_ < other.idx_;
    }

    return visit_n([](auto, const auto &l, const auto &r) { return l < r; },
                   *this, other);
  }

  // huh, clang format bug...
private : template <typename C> static constexpr void destroy_input(C &v) {
    v.C::~C();
  }

  unsigned idx_;
  tagged_union::detail::VariadicUnion<T...> union_;
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
