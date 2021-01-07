#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "meta/all_values.h"
#include "meta/container_concepts.h"
#include "meta/decays_to.h"
#include "meta/get_idx.h"
#include "meta/pack_element.h"
#include "meta/per_instance.h"
#include "meta/sequential_look_up.h"
#include "meta/specialization_of.h"
#include "meta/tag.h"

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
public:
  constexpr VariadicUnion() : first_{} {}

  constexpr ~VariadicUnion() requires(TriviallyDestructable<First, Rest...>) =
      default;

  // delegated to holder - must be manually destructed
  constexpr ~VariadicUnion() requires(!TriviallyDestructable<First, Rest...> &&
                                      Destructable<First, Rest...>) {}

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

template <AllValuesEnumerable E, std::movable... T>
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
      return sequential_look_up<values.size()>(first.idx_, [&](auto value) {
        constexpr unsigned idx = decltype(value)::value;
        return f(Tag<E, idx>{}, ac::get<idx>(vals.union_)...);
      });
    }(first, rest...);
  }

public:
  template <unsigned idx, typename... Args>
  requires(AggregateConstrucableFrom<Type<idx>, Args...>) constexpr TaggedUnion(
      Tag<E, idx>, Args &&...args)
      : idx_(idx),
        union_(std::in_place_index_t<idx>{}, std::forward<Args>(args)...) {}

  constexpr TaggedUnion() : TaggedUnion(Tag<E, 0>{}) {}

  constexpr TaggedUnion(const TaggedUnion &other) requires(
      TriviallyCopyConstructable<T...>) = default;

  constexpr TaggedUnion(const TaggedUnion &other) requires(
      !TriviallyCopyConstructable<T...> && CopyConstructable<T...>)
      : idx_(other.idx_) {
    visit_n(
        [](auto holder, auto &&l, auto &&r) {
          new (&l) PackElement<decltype(holder)::idx, T...>(r);
        },
        *this, other);
  }

  constexpr TaggedUnion(TaggedUnion &&other) requires(
      TriviallyMoveConstructable<T...>) = default;

  constexpr TaggedUnion(TaggedUnion &&other) requires(
      !TriviallyMoveConstructable<T...> && MoveConstructable<T...>)
      : idx_(other.idx_) {
    visit_n(
        [](auto holder, auto &&l, auto &&r) {
          new (&l) PackElement<decltype(holder)::idx, T...>(std::move(r));
        },
        *this, std::forward<TaggedUnion>(other));
  }

  constexpr TaggedUnion &operator=(const TaggedUnion &other) requires(
      TriviallyCopyAssignable<T...>) = default;

  constexpr TaggedUnion &
  operator=(const TaggedUnion &other) requires(!TriviallyCopyAssignable<T...> &&
                                               CopyAssignable<T...>) {
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

  constexpr TaggedUnion &operator=(TaggedUnion &&other) requires(
      TriviallyMoveAssignable<T...>) = default;

  constexpr TaggedUnion &
  operator=(TaggedUnion &&other) requires(!TriviallyMoveAssignable<T...> &&
                                          MoveAssignable<T...>) {
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

  constexpr ~TaggedUnion() requires(TriviallyDestructable<T...>) = default;

  constexpr ~TaggedUnion() requires(!TriviallyDestructable<T...> &&
                                    Destructable<T...>) {
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

private:
  template <typename C> static constexpr void destroy_input(C &v) { v.C::~C(); }

  unsigned idx_;
  tagged_union::detail::VariadicUnion<T...> union_;
};

namespace tagged_union {
namespace detail {
template <AllValuesEnumerable T, template <T> class TypeOver>
struct TaggedUnionPerInstanceImpl {
  template <typename... Ts> using Impl = TaggedUnion<T, Ts...>;

  using type = PerInstance<T, TypeOver, Impl>;
};
} // namespace detail
} // namespace tagged_union

template <AllValuesEnumerable T, template <T> class TypeOver>
using TaggedUnionPerInstance =
    typename tagged_union::detail::TaggedUnionPerInstanceImpl<T,
                                                              TypeOver>::type;

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
            return boost::hana::unpack(
                AllValues<PackElement<idx, Types...>>, [&](auto... v) {
                  return std::array<T, sizeof...(v)>{
                      T(TAG(tag_values[decltype(idx)::value]), v)...};
                });
          }(idx)...);

          static_assert(StdArraySpecialization<decltype(out)>);

          return out;
        });
  }();
};
