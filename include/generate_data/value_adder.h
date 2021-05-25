#pragma once

namespace generate_data {
// concatenates together float values using passed function
template <typename F> struct ValueAdder {
  F base_add;
  int idx = 0;

  void add_value(float v) { base_add(v, idx++); }

  template <typename T> void add_values(const T &vals) {
    for (const auto v : vals) {
      add_value(v);
    }
  }
};

template <typename F> auto make_value_adder(F v) { return ValueAdder<F>{v}; }
}; // namespace generate_data
