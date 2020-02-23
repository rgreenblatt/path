#pragma once

#include "compile_time_dispatch/compile_time_dispatch.h"

template <typename DispatchT, template <DispatchT> class TypeOver,
          unsigned idx = 0>
class OnePerInstance {
public:
  OnePerInstance(){};

  template <typename First, typename... Rest>
  OnePerInstance(const First &first, const Rest &... rest)
      : item_(first), next(rest...) {}

  static constexpr unsigned size = CompileTimeDispatchable<DispatchT>::size;

  static_assert(size != 0);

  static constexpr DispatchT this_value =
      CompileTimeDispatchable<DispatchT>::values[idx];

  using ItemType = TypeOver<this_value>;

  template <DispatchT value> const auto &get_item() const {
    if constexpr (value == this_value) {
      return item_;
    } else {
      static_assert(size != 1, "enum value not found");
      return next.template get_item<value>();
    }
  }

  template <DispatchT value> auto &get_item() {
    if constexpr (value == this_value) {
      return item_;
    } else {
      static_assert(size != 1, "enum value not found");
      return next.template get_item<value>();
    }
  }

private:
  ItemType item_;

  struct NoneType {};

  std::conditional_t<idx + 1 == size, NoneType,
                     OnePerInstance<DispatchT, TypeOver, idx + 1>>
      next;
};
