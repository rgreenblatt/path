#pragma once

#include "generate_data/remap_large.h"

#include <array>

namespace generate_data {
// concatenates together float values using passed function
template <typename F> struct ValueAdder {
  F base_add;
  int idx = 0;

  void add_value(float v) { base_add(v, idx++); }

  void add_remap_value(double v, double scale = 1e4) {
    add_value(remap_large(v, scale));
  }

  static constexpr std::array<double, 5> scales = {1e-2, 1e0, 1e3, 1e4, 1e5};

  void add_remap_multiscale_value(double v) {
    for (double scale : scales) {
      add_remap_value(v, scale);
    }
  }

  template <typename T> void add_values(const T &vals) {
    for (const auto v : vals) {
      add_value(v);
    }
  }

  template <typename T> void add_remap_values(const T &vals) {
    for (const auto v : vals) {
      add_remap_value(v);
    }
  }

  template <typename T> void add_remap_multiscale_values(const T &vals) {
    for (const auto v : vals) {
      add_remap_multiscale_value(v);
    }
  }

  void add_remap_all_value(double v) {
    add_value(v);
    add_remap_multiscale_value(v);
  }

  template <typename T> void add_remap_all_values(const T &vals) {
    for (const auto v : vals) {
      add_remap_all_value(v);
    }
  }
};

template <typename F> auto make_value_adder(F v) { return ValueAdder<F>{v}; }
}; // namespace generate_data
