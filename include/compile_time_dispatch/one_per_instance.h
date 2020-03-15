#pragma once

#include "compile_time_dispatch/compile_time_dispatch.h"

#include <iostream>

template <CompileTimeDispatchable T, template <T> class TypeOver,
          unsigned idx = 0>
class OnePerInstance {
public:
  OnePerInstance(){};

  template <typename First, typename... Rest>
  OnePerInstance(const First &first, const Rest &... rest)
      : item_(first), next(rest...) {}

  static constexpr unsigned size = CompileTimeDispatchableT<T>::size;

  static_assert(size != 0);

  static constexpr T this_value = CompileTimeDispatchableT<T>::values[idx];

  using ItemType = TypeOver<this_value>;

  template <T value> const auto &get_item() const {
    if constexpr (value == this_value) {
      return item_;
    } else {
      static_assert(size != 1, "dispatch value not found");
      return next.template get_item<value>();
    }
  }

  template <T value> auto &get_item() {
    if constexpr (value == this_value) {
      return item_;
    } else {
      static_assert(size != 1, "dispatch value not found");
      return next.template get_item<value>();
    }
  }

  template <typename F> auto visit(const F &f, const T &value) {
    if (value == this_value) {
      return f(item_);
    } else {
      if constexpr (std::is_same_v<decltype(next), std::tuple<>>) {
        std::cerr << "dispatch value not found" << std::endl;
        abort(); // maybe don't abort?
      } else {
        return next.visit(f, value);
      }
    }
  }

private:
  ItemType item_;

  std::conditional_t<idx + 1 == size, std::tuple<>,
                     OnePerInstance<T, TypeOver, idx + 1>>
      next;
};
