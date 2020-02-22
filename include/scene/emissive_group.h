#pragma once

#include "intersect/accel/aabb.h"

namespace scene {
struct EmissiveGroup {
  unsigned material_idx;
  unsigned start_idx;
  unsigned end_idx;
  intersect::accel::AABB aabb;
};
} // namespace scene
