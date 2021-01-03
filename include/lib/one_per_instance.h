#pragma once

#include "meta/all_values.h"
#include "meta/get_idx.h"
#include "meta/sequential_look_up.h"
#include "meta/per_instance.h"

#include <compare>
#include <tuple>
#include <utility>

template <AllValuesEnumerable T, template <T> class TypeOver>
class OnePerInstance {
public:
  using Items = PerInstance<T, TypeOver, std::tuple>;

  constexpr OnePerInstance(){};

  constexpr OnePerInstance(const Items &items) : items_(items){};

  template <T value> ATTR_PURE_NDEBUG constexpr const auto &get() const {
    return std::get<get_idx(value)>(items_);
  }

  template <T value> ATTR_PURE_NDEBUG constexpr auto &get() {
    return std::get<get_idx(value)>(items_);
  }

  template <T value> using Tag = Tag<T, value>;

  template <T value>
  ATTR_PURE_NDEBUG constexpr const auto &get(Tag<value>) const {
    return get<value>();
  }

  template <T value> ATTR_PURE_NDEBUG constexpr auto &get(Tag<value>) {
    return get<value>();
  }

  template <typename F> constexpr auto visit(const F &f, const T &value) {
    return sequential_look_up<AllValues<T>.size()>(get_idx(value), [&](auto idx) {
      return f(std::get<decltype(idx)::value>(items_));
    });
  }

  constexpr bool operator==(const OnePerInstance &other) const = default;
  constexpr auto operator<=>(const OnePerInstance &other) const = default;

private:
  Items items_;
};
