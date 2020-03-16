#pragma once

#include "compile_time_dispatch/compile_time_dispatch.h"
#include "compile_time_dispatch/to_array.h"

#include <petra/sequential_table.hpp>

#include <map>

template <CompileTimeDispatchable T, std::size_t idx> struct Holder {
  static constexpr auto value = CompileTimeDispatchableT<T>::values[idx];
};

template<typename>
struct Printer;

template <typename F, CompileTimeDispatchable T>
auto dispatch_value(const F &f, T value) {
  using Dispatch = CompileTimeDispatchableT<T>;

  static_assert(Dispatch::size != 0);
    
  using ValT = std::decay_t<decltype(Dispatch::values[0])>;

  auto get_result =
      petra::make_sequential_table<Dispatch::size - 1>([&](auto &&i) {
        using PetraT = decltype(i);
        if constexpr (petra::utilities::is_error_type<PetraT>()) {
          std::cerr << "invalid dispatch value!" << std::endl;
          abort();
        } else {
          constexpr std::size_t index = std::decay_t<decltype(i)>::value;

          return f(Holder<ValT, index>{});
        }
      });

  const static auto lookup = [] {
    std::map<ValT, std::size_t> lookup;

    static_assert(Dispatch::size == Dispatch::values.size());
    for (std::size_t i = 0; i < Dispatch::size; i++) {
      lookup.insert(std::pair{Dispatch::values[i], i});
    }

    return lookup;
  }();

  auto it = lookup.find(value);
  if (it == lookup.end()) {
    std::cerr << "invalid dispatch value!" << std::endl;
    abort();
  }

  return get_result(it->second);
}
