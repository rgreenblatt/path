#pragma once

#include "lib/compile_time_dispatch/compile_time_dispatch.h"
#include "lib/compile_time_dispatch/to_array.h"

#include <petra/sequential_table.hpp>

#include <map>

template <typename T, unsigned idx> struct Holder {
  static constexpr auto value = CompileTimeDispatchable<T>::values[idx];
};

template <typename F, typename T,
          typename Dispatch = CompileTimeDispatchable<T>>
auto dispatch_value(const F &f, T value) {
  static_assert(Dispatch::size != 0);

  auto get_result =
      petra::make_sequential_table<Dispatch::size - 1>([&](auto &&i) {
        using PetraT = decltype(i);
        if constexpr (petra::utilities::is_error_type<PetraT>()) {
          std::cerr << "invalid dispatch value!" << std::endl;
          abort();
        } else {
          constexpr unsigned index = std::decay_t<decltype(i)>::value;

          return f(Holder<T, index>{});
        }
      });

  const static std::map<T, unsigned> lookup = [] {
    std::map<T, unsigned> lookup;

    for (unsigned i = 0; i < Dispatch::size; i++) {
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
