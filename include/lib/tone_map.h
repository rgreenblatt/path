#pragma once

#include "lib/attribute.h"

// could be impure depending on T
template <typename T>
ATTR_NO_DISCARD_PURE constexpr decltype(auto) tone_map(const T &v) {
  return v / (v + 1);
}
