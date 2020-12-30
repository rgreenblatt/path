#pragma once

#include "meta/all_values.h"
#include "meta/get_idx.h"
#include "meta/sequential_look_up.h"

#include <compare>
#include <tuple>
#include <utility>

template <AllValuesEnumerable T, template <T> class TypeOver>
class OnePerInstance {
private:
  static constexpr auto values = AllValues<T>;
  static constexpr unsigned size = AllValues<T>.size();

  template <std::size_t... i>
  static constexpr auto items_helper(std::integer_sequence<std::size_t, i...>) {
    return std::tuple<TypeOver<values[i]>...>{};
  }

public:
  using Items =
      decltype(OnePerInstance::items_helper(std::make_index_sequence<size>{}));

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
    return sequential_look_up<size>(get_idx(value), [&](auto idx) {
      return f(std::get<decltype(idx)::value>(items_));
    });
  }

  constexpr bool operator==(const OnePerInstance &other) const = default;
  constexpr auto operator<=>(const OnePerInstance &other) const = default;

private:
  Items items_;
};
