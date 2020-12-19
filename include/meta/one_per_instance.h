#pragma once

#include "meta/all_values.h"
#include "meta/sequential_look_up.h"

#include <iostream>

template <AllValuesEnumerable T, template <T> class TypeOver>
class OnePerInstance {
private:
  static constexpr auto values = AllValues<T>;
  static constexpr unsigned size = AllValues<T>.size();

  template<std::size_t...i>
  static constexpr auto items_helper(std::integer_sequence<std::size_t, i...>) {
    return std::tuple<TypeOver<values[i]>...>{};
  }

public:
  using Items =
      decltype(OnePerInstance::items_helper(std::make_index_sequence<size>{}));

  OnePerInstance(){};

  OnePerInstance(const Items &items) : items_(items){};

  template <T value> const auto &get_item() const {
    constexpr unsigned idx = OnePerInstance::get_idx(value);

    return std::get<idx>(items_);
  }

  template <T value> auto &get_item() {
    constexpr unsigned idx = OnePerInstance::get_idx(value);

    return std::get<idx>(items_);
  }

  template <typename F> auto visit(const F &f, const T &value) {
    unsigned idx = OnePerInstance::get_idx(value);
    return sequential_look_up<size>(idx, [&](auto idx) {
      return f(std::get<decltype(idx)::value>(items_));
    });
  }

private:
  static constexpr unsigned get_idx(const T& value) {
    for (unsigned i = 0; i < size; ++i) {
      if (values[i] == value) {
        return i;
      }
    }
    std::cerr << "failed to lookup value" << std::endl;
    assert(false);
    abort();
  }
  Items items_;
};
