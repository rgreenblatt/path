#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/mesh_instance.h"
#include "intersect/triangle.h"
#include "lib/span.h"
#include "material/material.h"
#include "scene/triangle_data.h"

#include <thrust/optional.h>
#include <tiny_obj_loader.h>

#include <vector>

namespace scene {
namespace scenefile_compat {
class ScenefileLoader;
}

class Scene {
public:
  const Eigen::Affine3f &film_to_world() const { return film_to_world_; }

  using Triangle = intersect::Triangle;
  using MeshInstance = intersect::MeshInstance;

  SpanSized<const unsigned> mesh_ends() const { return mesh_ends_; }

  SpanSized<const intersect::accel::AABB> mesh_aabbs() const {
    return mesh_aabbs_;
  }

  SpanSized<const std::string> mesh_paths() const { return mesh_paths_; }

  SpanSized<const MeshInstance> mesh_instances() const {
    return mesh_instances_;
  }

  SpanSized<const Triangle> triangles() const { return triangles_; }

  SpanSized<const TriangleData> triangle_data() const { return triangle_data_; }

  SpanSized<const material::Material> materials() const { return materials_; }

  // Note: may not be very precise...
  intersect::accel::AABB overall_aabb() const { return overall_aabb_; }

#if 0
  SpanSized<const CS123SceneLightData> lights() const { return lights_; }
#endif

private:
  Scene() {}

  std::vector<unsigned> mesh_ends_;
  std::vector<intersect::accel::AABB> mesh_aabbs_; // not transformed
  std::vector<std::string> mesh_paths_;            // used as unique identifiers
  std::vector<MeshInstance> mesh_instances_;
  std::vector<Triangle> triangles_;
  std::vector<TriangleData> triangle_data_;
  std::vector<material::Material> materials_;

  intersect::accel::AABB overall_aabb_;

#if 0
  std::vector<CS123SceneLightData> lights_;
#endif

  Eigen::Affine3f film_to_world_;

  friend class scenefile_compat::ScenefileLoader;
};
} // namespace scene
