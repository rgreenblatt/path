#pragma once

// compare to std::integral constant
template <unsigned idx_in> struct NTag {
  static constexpr unsigned idx = idx_in;
};
