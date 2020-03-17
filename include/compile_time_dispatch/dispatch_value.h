#pragma once

#include "compile_time_dispatch/compile_time_dispatch.h"
#include "compile_time_dispatch/to_array.h"

#include <petra/sequential_table.hpp>

#include <map>

template <CompileTimeDispatchable T, std::size_t idx> struct Holder {
  static_assert(idx < CompileTimeDispatchableT<T>::values.size());
  static constexpr auto value =
      std::get<idx>(CompileTimeDispatchableT<T>::values);
};

template <typename F, CompileTimeDispatchable T>
auto dispatch_value(const F &f, T value) {
  using Dispatch = CompileTimeDispatchableT<T>;

  static_assert(Dispatch::size != 0);

  using ValT = std::decay_t<decltype(Dispatch::values[0])>;

  // Petra bug...
  auto get_result = petra::make_sequential_table<Dispatch::size>([&](auto &&i) {
    using PetraT = decltype(i);
    if constexpr (petra::utilities::is_error_type<PetraT>()) {
      std::cerr << "Internal dispatch error (petra err)" << std::endl;
      abort();
    } else {
      constexpr std::size_t index = std::decay_t<decltype(i)>::value;
      if constexpr (index >= Dispatch::size) {
        std::cerr << "Internal dispatch error (index over)" << std::endl;
        abort();
      } else {
        return f(Holder<ValT, index>{});
      }
    }
  });

  const static auto lookup = [] {
    std::map<ValT, std::size_t> lookup;

    static_assert(Dispatch::size == Dispatch::values.size());
    for (std::size_t i = 0; i < Dispatch::size; i++) {
      lookup.insert(std::pair{Dispatch::values[i], i});
    }

    // fails if not all dispatch values are unique
    assert(lookup.size() == Dispatch::size);

    return lookup;
  }();

  auto it = lookup.find(value);
  if (it == lookup.end()) {
    std::cerr << "invalid dispatch value!" << std::endl;
    abort();
  }

  return get_result(it->second);
}
