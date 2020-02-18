#pragma once

#include "scene/scene.h"
#include "scene/scenefile_compat/CS123SceneData.h"

#include <map>
#include <string>

namespace scene {
namespace scenefile_compat {
class ScenefileLoader {
public:
  inline thrust::optional<Scene> load_scene(const std::string &filename,
                                            float width_height_ratio);

private:
  bool parse_tree(Scene &scene, const CS123SceneNode &root,
                  const std::string &base_dir);

  bool parse_node(Scene &scene, const CS123SceneNode &node,
                  const Eigen::Affine3f &parent_transform,
                  const std::string &base_dir);

  bool add_primitive(Scene &scene, const CS123ScenePrimitive &prim,
                     const Eigen::Affine3f &transform,
                     const std::string &base_dir);

  bool load_mesh(Scene &scene, std::string file_path,
                 const Eigen::Affine3f &transform, const std::string &base_dir);

  bool add_light(Scene &scene, const CS123SceneLightData &data);

  std::map<std::string, unsigned> loaded_meshes_; // avoid reloading dup meshes

  CS123SceneGlobalData global_data_; // TODO: what is global data for....
};
} // namespace scenefile_compat
} // namespace scene
