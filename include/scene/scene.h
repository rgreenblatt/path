#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/triangle.h"
#include "lib/span.h"
#include "lib/span_convertable_vector.h"
#include "scene/CS123SceneData.h"
#include "scene/triangle_data.h"

#include <thrust/optional.h>
#include <tiny_obj_loader.h>

#include <map>
#include <memory>
#include <vector>

namespace scene {
struct MeshInstance {
  unsigned idx;
  Eigen::Affine3f transform;
};

class Scene {
public:
  static thrust::optional<Scene> make_scene(const std::string &filename,
                                            float width_height_ratio);
  virtual ~Scene();

  const Eigen::Affine3f &film_to_world() const;

  using Triangle = intersect::Triangle;

  SpanSized<const CS123SceneLightData> lights() const { return lights_; }

  SpanSized<const unsigned> mesh_ends() const { return mesh_ends_; }

  SpanSized<const MeshInstance> mesh_instances() const {
    return mesh_instances_;
  }

  SpanSized<const Triangle> triangles() const { return triangles_; }

  SpanSized<const TriangleData> triangle_data() const { return triangle_data_; }

  SpanSized<const tinyobj::material_t> materials() const { return materials_; }

private:
  std::vector<unsigned> mesh_ends_;
  std::vector<intersect::accel::AABB> mesh_aabbs_;
  std::vector<MeshInstance> mesh_instances_;
  std::vector<Triangle> triangles_;
  std::vector<TriangleData> triangle_data_;
  std::vector<tinyobj::material_t> materials_;

  std::vector<CS123SceneLightData> lights_;

  std::map<std::string, unsigned> loaded_meshes_; // avoid reloading dup meshes

  Eigen::Affine3f film_to_world_;

  CS123SceneGlobalData global_data_;

  bool parse_tree(const CS123SceneNode &root, const std::string &base_dir);

  bool parse_node(const CS123SceneNode &node,
                  const Eigen::Affine3f &parent_transform,
                  const std::string &base_dir);

  bool add_primitive(const CS123ScenePrimitive &prim,
                     const Eigen::Affine3f &transform,
                     const std::string &base_dir);

  bool load_mesh(std::string file_path, const Eigen::Affine3f &transform,
                 const std::string &base_dir);

  void add_light(const CS123SceneLightData &data);
};
} // namespace scene
