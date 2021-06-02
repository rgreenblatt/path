#include "scene/triangle_constructor.h"

namespace scene {
unsigned TriangleConstructor::add_material(const Material &material) {
  unsigned out = scene_.materials_.size();
  scene_.materials_.push_back(material);
  material_is_emissive_.push_back(material.emission().matrix().squaredNorm() >
                                  1e-9f);

  return out;
}

void TriangleConstructor::add_triangle(const intersect::Triangle &triangle,
                                       unsigned int material_idx) {
  auto normal = triangle.normal();
  std::array<UnitVector, 3> normals{normal, normal, normal};
  unsigned start_idx = scene_.triangles_.size();
  scene_.triangles_.push_back_all(triangle, {normals, material_idx});

  if (material_is_emissive_[material_idx]) {
    scene_.emissive_clusters_.push_back({
        .material_idx = material_idx,
        .start_idx = start_idx,
        .end_idx = scene_.triangles_.size(),
        .aabb = triangle.bounds(),
    });
  }
}

const Scene &TriangleConstructor::scene(const std::string &mesh_name) {
  using intersect::accel::AABB;

  if (scene_.meshs_.empty()) {
    AABB overall_aabb = AABB::empty();

    for (const intersect::Triangle &tri :
         scene_.triangles_.get(tag_v<Scene::TriangleT::Inst>)) {
      overall_aabb = overall_aabb.union_other(tri.bounds());
    }

    scene_.meshs_.push_back_all(scene_.triangles_.size(), overall_aabb,
                                mesh_name, scene_.emissive_clusters_.size());
    scene_.transformed_objects_.push_back_all(
        intersect::TransformedObject(Eigen::Affine3f::Identity(), overall_aabb),
        0);
  }

  always_assert(scene_.meshs_.size() == 1 &&
                scene_.transformed_objects_.size() == 1);
  always_assert(scene_.meshs_.get(tag_v<Scene::MeshT::End>)[0] ==
                scene_.triangles_.size());
  return scene_;
}
} // namespace scene
