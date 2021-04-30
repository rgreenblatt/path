#pragma once

#include "intersect/accel/accel.h"
#include "intersect/triangle.h"

namespace intersect {
namespace accel {
template <typename T, typename Settings>
concept TriangleAccel = ObjectSpecificAccel<T, Settings, Triangle>;
}
} // namespace intersect
