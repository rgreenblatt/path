#pragma once

#include "lib/optional.h"
#include "scene/scene.h"
#include "scene/scenefile_compat/CS123SceneData.h"

#include <map>
#include <string>

namespace scene {
namespace scenefile_compat {
class ScenefileLoader {
public:
  Optional<Scene> load_scene(const std::string &filename,
                             float width_height_ratio, bool quiet = false);

private:
  bool parse_tree(Scene &scene_v, const CS123SceneNode &root,
                  const std::string &base_dir);

  bool parse_node(Scene &scene_v, const CS123SceneNode &node,
                  const Eigen::Affine3f &parent_transform,
                  const std::string &base_dir);

  bool add_primitive(Scene &scene_v, const CS123ScenePrimitive &prim,
                     const Eigen::Affine3f &transform,
                     const std::string &base_dir);

  bool load_mesh(Scene &scene_v, std::string file_path,
                 const Eigen::Affine3f &transform, const std::string &base_dir);

  bool add_light(Scene &scene_v, const CS123SceneLightData &data);

  bool quiet_ = false;

  std::map<std::string, unsigned> loaded_meshes_; // avoid reloading dup meshes

  std::vector<uint8_t> is_emissive;

  CS123SceneGlobalData global_data_; // TODO: what is global data for....

  Eigen::Vector3f overall_min_b_transformed_;
  Eigen::Vector3f overall_max_b_transformed_;
};
} // namespace scenefile_compat
} // namespace scene
