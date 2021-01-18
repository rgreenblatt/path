#pragma once

#include "lib/assert.h"
#include "meta/all_values.h"
#include "meta/sequential_dispatch.h"
#include "meta/tag.h"
#include "meta/tag_dispatchable.h"

#include <iostream>
#include <map>

// this probably isn't a very efficient implementation (double dispatch...)
template <AllValuesEnumerable T, TagDispatchable<T> F>
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
