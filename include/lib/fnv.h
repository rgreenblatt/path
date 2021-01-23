#pragma once

#include "lib/attribute.h"

ATTR_PURE_NDEBUG constexpr unsigned fnv_hash(unsigned in) {
  constexpr unsigned prime = 16777619;
  constexpr unsigned offset_basis = 2166136261;
  unsigned out_hash = offset_basis;
  for (unsigned i = 0; i < 4; ++i) {
    out_hash ^= (in >> (8 * i)) & 0xff;
    out_hash *= prime;
  }
  return out_hash;
}
