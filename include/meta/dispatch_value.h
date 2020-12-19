#pragma once

#include "meta/all_values.h"
#include "meta/sequential_look_up.h"

#include <map>

template <AllValuesEnumerable T, unsigned idx> struct Holder {
  static_assert(idx < AllValues<T>.size());
  static constexpr auto value = AllValues<T>[idx];
};

template <typename F, AllValuesEnumerable T>
auto dispatch_value(const F &f, T value) {
  const static auto lookup = [] {
    constexpr auto values = AllValues<T>;

    static_assert(values.size() != 0);

    std::map<T, unsigned> lookup;

    for (unsigned i = 0; i < values.size(); i++) {
      lookup.insert(std::pair{values[i], i});
    }

    // fails if not all dispatch values are unique
    if (lookup.size() != values.size()) {
      std::cerr << "internal dispatch error (not all values unique)"
                << std::endl;
      assert(false);
      abort();
    }

    return lookup;
  }();

  auto it = lookup.find(value);
  if (it == lookup.end()) {
    std::cerr << "invalid dispatch value!" << std::endl;
    assert(false);
    abort();
  }

  return sequential_look_up<AllValues<T>.size()>(
      it->second, [&](auto i) { return f(Holder<T, decltype(i)::value>{}); });
}
