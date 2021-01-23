#pragma once

#include "lib/float_rgb.h"

namespace cereal {
template <typename Archive> void serialize(Archive &ar, FloatRGB &arr) {
  ar(arr[0], arr[1], arr[2]);
}
} // namespace cereal
