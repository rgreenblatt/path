#pragma once

#include "meta/all_values.h"
#include "meta/concepts.h"
#include "meta/get_idx.h"
#include "meta/sequential_look_up.h"

#include <iostream>

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

  OnePerInstance(){};

  OnePerInstance(const Items &items) : items_(items){};

  template <T value> const auto &get() const {
    return std::get<get_idx(value)>(items_);
  }

  template <T value> auto &get() { return std::get<get_idx(value)>(items_); }

  template <T value> using Tag = Tag<T, value>;

  template <T value> const auto &get(Tag<value>) const { return get<value>(); }

  template <T value> auto &get(Tag<value>) { return get<value>(); }

  template <typename F> auto visit(const F &f, const T &value) {
    return sequential_look_up<size>(get_idx(value), [&](auto idx) {
      return f(std::get<decltype(idx)::value>(items_));
    });
  }

private:
  Items items_;
};
