#pragma once

#include <boost/hana/unpack.hpp>

template <typename T, typename V> constexpr T unpack_to(V &&v) {
  return boost::hana::unpack(v, [](auto &&...vals) -> T { return {vals...}; });
}
