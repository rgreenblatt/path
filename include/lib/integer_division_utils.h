#pragma once

#include "lib/attribute.h"

ATTR_PURE constexpr unsigned ceil_divide(unsigned l, unsigned r) {
  return (l + r - 1) / r;
};

ATTR_PURE constexpr unsigned make_divisable(unsigned v, unsigned divisable_by) {
  return ceil_divide(v, divisable_by) * divisable_by;
}
