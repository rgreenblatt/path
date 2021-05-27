#pragma once

#include "lib/attribute.h"

#include <cmath>

namespace generate_data {
ATTR_PURE_NDEBUG inline double remap_large(double v) {
  return std::tanh(v / 1e4);
}
} // namespace generate_data
