#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/transformed_object.h"
#include "intersect/triangle.h"
#include "lib/span.h"
#include "material/material.h"
#include "scene/emissive_group.h"
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
  using TransformedObject = intersect::TransformedObject;

  SpanSized<const unsigned> mesh_ends() const { return mesh_ends_; }

  SpanSized<const intersect::accel::AABB> mesh_aabbs() const {
    return mesh_aabbs_;
  }

  SpanSized<const std::string> mesh_paths() const { return mesh_paths_; }

  SpanSized<const TransformedObject> transformed_mesh_objects() const {
    return transformed_mesh_objects_;
  }

  SpanSized<const unsigned> transformed_mesh_idxs() const {
    return transformed_mesh_idxs_;
  }


  SpanSized<const Triangle> triangles() const { return triangles_; }

  SpanSized<const TriangleData> triangle_data() const { return triangle_data_; }

  SpanSized<const material::Material> materials() const { return materials_; }

  SpanSized<const EmissiveGroup> emissive_groups() const {
    return emissive_groups_;
  }

  SpanSized<const unsigned> emissive_group_ends_per_mesh() const {
    return emissive_group_ends_per_mesh_;
  }

  // Note: may not be very precise...
  intersect::accel::AABB overall_aabb() const { return overall_aabb_; }

private:
  Scene() {}

  std::vector<unsigned> mesh_ends_;
  std::vector<intersect::accel::AABB> mesh_aabbs_; // not transformed
  std::vector<std::string> mesh_paths_;            // used as unique identifiers
  std::vector<TransformedObject> transformed_mesh_objects_;
  std::vector<unsigned> transformed_mesh_idxs_;
  std::vector<Triangle> triangles_;
  std::vector<TriangleData> triangle_data_;
  std::vector<material::Material> materials_;
  std::vector<EmissiveGroup> emissive_groups_;
  std::vector<unsigned> emissive_group_ends_per_mesh_;

  intersect::accel::AABB overall_aabb_;

#if 0
  std::vector<CS123SceneLightData> lights_;
#endif

  Eigen::Affine3f film_to_world_;

  friend class scenefile_compat::ScenefileLoader;
};
} // namespace scene
