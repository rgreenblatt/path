#pragma once

#include "lib/attribute.h"

#include <cmath>

namespace generate_data {
ATTR_PURE_NDEBUG inline double remap_large(double v, double scale = 1e3) {
  return std::tanh(v / scale);
}
} // namespace generate_data
