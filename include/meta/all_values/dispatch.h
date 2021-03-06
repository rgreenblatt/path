#pragma once

#include "lib/assert.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/sequential_dispatch.h"
#include "meta/all_values/tag.h"

#include <iostream>
#include <map>

// this probably isn't a very efficient implementation (double dispatch...)
template <AllValuesEnumerable T, typename F>
requires(AllValues<T>.size() != 0) decltype(auto) dispatch(T value, F &&f) {
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

  return sequential_dispatch<AllValues<T>.size()>(
      it->second, [&]<unsigned idx>(NTag<idx>) { return f(Tag<T, idx>{}); });
}

// useful for type checking/writing code in some cases
template <AllValuesEnumerable T, typename F>
requires(AllValues<T>.size() != 0)
    // deprecated to warn on usage...
    [[deprecated(
        "remember to remove this use of fake_dispatch!")]] decltype(auto)
        fake_dispatch(T, F &&f) {
  return f(Tag<T, 0>{});
}
