#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "meta/aggregate_constructable_from.h"
#include "meta/all_values.h"
#include "meta/decays_to.h"
#include "meta/get_idx.h"
#include "meta/per_instance.h"
#include "meta/sequential_look_up.h"
#include "meta/specialization_of.h"

#include <boost/hana/fold_left.hpp>
#include <boost/hana/unpack.hpp>
#include <magic_enum.hpp>

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

  // delegated to holder - must be manually destructed
  constexpr ~VariadicUnion() {}

  template <std::size_t idx, typename... Args>
  requires(idx <=
           sizeof...(Rest)) constexpr VariadicUnion(std::in_place_index_t<idx>,
                                                    Args &&...args)
      : rest_(std::in_place_index_t<idx - 1>{}, std::forward<Args>(args)...) {}

  template <typename... Args>
  constexpr VariadicUnion(std::in_place_index_t<std::size_t(0)>, Args &&...args)
      : first_{std::forward<Args>(args)...} {}

private:
  First first_;
  VariadicUnion<Rest...> rest_;

  friend struct access;
};

struct access {
  template <std::size_t idx, SpecializationOf<VariadicUnion> UnionType>
  static constexpr auto &&get(UnionType &&v) {
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

  template <E tag> using Tag = Tag<E, tag>;

  using ac = tagged_union::detail::access;

  template <E type> using Type = __type_pack_element<get_idx(type), T...>;

  template <DecaysTo<TaggedUnion>... Ts, typename F>
  static constexpr decltype(auto) visit_n_at_idx(unsigned idx, F &&f,
                                                 Ts &&...vals) {
    return sequential_look_up<values.size()>(idx, [&](auto value) {
      constexpr unsigned idx = decltype(value)::value;
      return f(ac::get<idx>(vals.union_)...);
    });
  }

  template <DecaysTo<TaggedUnion> First, DecaysTo<TaggedUnion>... Rest,
            typename F>
  static constexpr decltype(auto) visit_n(F &&f, First &&first,
                                          Rest &&...rest) {
    debug_assert(((first.idx_ == rest.idx_) && ... && true));
    return visit_n_at_idx(first.idx_, f, first, rest...);
  }

public:
  template <E tag, typename... Args>
  requires(AggregateConstrucableFrom<Type<tag>, Args...>) constexpr TaggedUnion(
      Tag<tag>, Args &&...args)
      : idx_(get_idx(tag)), union_(std::in_place_index_t<get_idx(tag)>{},
                                   std::forward<Args>(args)...) {}

  constexpr TaggedUnion() : TaggedUnion(Tag<values[0]>{}) {}

  constexpr ~TaggedUnion() {
    visit([](auto &in) { destroy_input(in); });
  }

  constexpr TaggedUnion(TaggedUnion &&other) : idx_(other.idx_) {
    visit_n([](auto &&l, auto &&r) { l = std::move(r); }, *this,
            std::forward<TaggedUnion>(other));
  }

  constexpr TaggedUnion(const TaggedUnion &other) : idx_(other.idx_) {
    visit_n([](auto &&l, auto &&r) { l = r; }, *this, other);
  }

  constexpr TaggedUnion &operator=(const TaggedUnion &other) {
    if (this != &other) {
      this->TaggedUnion::~TaggedUnion();
      idx_ = other.idx_;
      visit_n([](auto &&l, auto &&r) { l = r; }, *this, other);
    }
    return *this;
  }

  constexpr TaggedUnion &operator=(TaggedUnion &&other) {
    if (this != &other) {
      this->TaggedUnion::~TaggedUnion();
      idx_ = other.idx_;
      visit_n([](auto &&l, auto &&r) { l = std::move(r); }, *this,
              std::forward<TaggedUnion>(other));
    }
    return *this;
  }

  template <E type, typename... Args>
  constexpr static TaggedUnion create(Args &&...args) {
    return TaggedUnion(Tag<type>{}, std::forward<Args>(args)...);
  }

  ATTR_PURE_NDEBUG constexpr E type() const { return values[idx_]; }

  template <typename F> constexpr decltype(auto) visit(F &&f) const {
    return visit_n(std::forward<F>(f), *this);
  }

  template <typename F> constexpr decltype(auto) visit(F &&f) {
    return visit_n(std::forward<F>(f), *this);
  }

  template <typename F> constexpr decltype(auto) visit_indexed(F &&f) {
    return visit([&](auto &&v) { return f(type(), v); });
  }

  template <typename F> constexpr decltype(auto) visit_indexed(F &&f) const {
    return visit([&](auto &&v) { return f(type(), v); });
  }

  template <E type> constexpr auto &get() {
    debug_assert_assume(get_idx(type) == idx_);
    return ac::get<get_idx(type)>(union_);
  }

  template <E type> constexpr const auto &get() const {
    debug_assert_assume(get_idx(type) == idx_);
    return ac::get<get_idx(type)>(union_);
  }

  constexpr auto operator<=>(const TaggedUnion &other) const
      requires(... &&std::totally_ordered<T>) {
    if (idx_ != other.idx_) {
      return idx_ <=> other.idx_;
    }

    return visit_n([](const auto &l, const auto &r) { return l <=> r; }, *this,
                   other);
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
  template<typename...Ts>
  using Impl = TaggedUnion<T, Ts...>;

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

public:
  static constexpr auto values = [] {
    return boost::hana::fold_left(
        std::make_index_sequence<num_elements>(), std::array<T, 0>{},
        [](auto arr, auto idx) {
          return std::tuple_cat(
              arr, boost::hana::unpack(
                       AllValues<__type_pack_element<idx, T>>, [&](auto... v) {
                         return std::array{
                             T::T<magic_enum::enum_value<E>(idx)>(v)...};
                       }));
        });
  }();
};
