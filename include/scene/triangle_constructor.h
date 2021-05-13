#pragma once

#include "lib/vector_type.h"
#include "scene/scene.h"

namespace scene {
// this is mostly for testing
class TriangleConstructor {
public:
  // returns idx
  unsigned add_material(const Material &material);

  void add_triangle(const intersect::Triangle &triangle, unsigned material_idx);

  const Scene &scene(const std::string &mesh_name);

private:
  Scene scene_;
  VectorT<uint8_t> material_is_emissive_;
};
} // namespace scene
