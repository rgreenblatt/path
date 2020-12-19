#pragma once

#include "meta/all_values.h"
#include "meta/to_array.h"

#include <petra/sequential_table.hpp>

#include <map>

template <AllValuesEnumerable T, std::size_t idx> struct Holder {
  static_assert(idx < AllValues<T>.size());
  static constexpr auto value = AllValues<T>[idx];
};

template <typename F, AllValuesEnumerable T>
auto dispatch_value(const F &f, T value) {
  constexpr auto values = AllValues<T>;

  static_assert(values.size() != 0);

  // Petra bug...
  auto get_result = petra::make_sequential_table<values.size()>([&](auto &&i) {
    using PetraT = decltype(i);
    if constexpr (petra::utilities::is_error_type<PetraT>()) {
      std::cerr << "Internal dispatch error (petra err)" << std::endl;
      abort();
    } else {
      constexpr std::size_t index = std::decay_t<decltype(i)>::value;
      if constexpr (index >= values.size()) {
        std::cerr << "Internal dispatch error (index over)" << std::endl;
        abort();
      } else {
        return f(Holder<T, index>{});
      }
    }
  });

  const static auto lookup = [] {
    std::map<T, std::size_t> lookup;

    for (std::size_t i = 0; i < values.size(); i++) {
      lookup.insert(std::pair{values[i], i});
    }

    // fails if not all dispatch values are unique
    if (lookup.size() != values.size()) {
      std::cerr << "internal dispatch error (not all values unique)"
                << std::endl;
      abort();
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
