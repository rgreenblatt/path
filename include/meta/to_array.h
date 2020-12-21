#pragma once

#include <boost/hana/unpack.hpp>

#include <array>

template <typename Vals> constexpr auto to_array(const Vals &vals) {
  return boost::hana::unpack(vals,
                             [](auto &&...x) { return std::array{x...}; });
}
