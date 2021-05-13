#pragma once

#include "generate_data/scene_triangles.h"
#include "lib/attribute.h"
#include "lib/unit_vector.h"

namespace generate_data {
template <typename T>
ATTR_PURE_NDEBUG inline UnitVectorGen<T>
get_dir_towards(const SceneTrianglesGen<T> &tris) {
  auto onto_normal = tris.triangle_onto.normal();
  auto onto_point = tris.triangle_onto.vertices[0];
  auto light_centroid = tris.triangle_light.centroid();

  if (onto_normal->dot(light_centroid - onto_point) > 0.) {
    // positive side of plane
    return UnitVectorGen<T>::new_unchecked(-(*onto_normal));
  } else {
    // negative side of plane
    return onto_normal;
  }
}

} // namespace generate_data
