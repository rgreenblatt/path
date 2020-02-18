#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"

namespace intersect {
template <typename AccelPrim>
// TODO check if ignore v is needed (probably not)
// Also, use indices or something??
inline HOST_DEVICE IntersectionOp<GlobalLocation>
solve(const Ray &world_space_ray, const AccelPrim &accel_prim) {
  return accel_prim(world_space_ray);
}
} // namespace intersect
