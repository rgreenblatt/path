#pragma once

#include "lib/assert.h"
#include "meta/all_values.h"
#include "meta/sequential_look_up.h"
#include "meta/tag.h"

#include <iostream>
#include <map>

template <typename F, AllValuesEnumerable T>
requires (AllValues<T>.size() != 0)
auto dispatch_value(const F &f, T value) {
  const static auto lookup = [] {
    constexpr auto values = AllValues<T>;

    std::map<T, unsigned> lookup;

    for (unsigned i = 0; i < values.size(); i++) {
      lookup.insert(std::pair{values[i], i});
    }

    // fails if not all dispatch values are unique
    if (lookup.size() != values.size()) {
      std::cerr << "internal dispatch error (not all values unique)"
                << std::endl;
      unreachable();
    }

    return lookup;
  }();

  auto it = lookup.find(value);
  if (it == lookup.end()) {
    std::cerr << "invalid dispatch value!" << std::endl;
    unreachable();
  }

  return sequential_look_up<AllValues<T>.size()>(
      it->second, [&](auto i) { return f(Tag<T, decltype(i)::value>{}); });
}
