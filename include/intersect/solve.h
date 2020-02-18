#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"

namespace intersect {
template <typename AccelPrim, typename AccelMesh>
// TODO check if ignore v is needed (probably not)
// Also, use indices or something??
inline HOST_DEVICE IntersectionOp<GlobalLocation>
solve(const AccelPrim &accel_prim, const AccelMesh &accel_mesh,
      const Ray &world_space_ray) {
  return accel_prim(world_space_ray, accel_mesh);
}
} // namespace intersect
