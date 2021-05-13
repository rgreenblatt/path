#pragma once

#include "lib/optional.h"
#include "scene/scene.h"
#include "scene/scenefile_compat/CS123SceneData.h"

#include <map>
#include <optional>
#include <string>

namespace scene {
namespace scenefile_compat {
struct SceneCamera {
  Scene scene;
  Eigen::Affine3f film_to_world;
};

class ScenefileLoader {
public:
  std::optional<SceneCamera> load_scene(const std::string &filename,
                                        float width_height_ratio,
                                        bool quiet = false);

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

  bool quiet_ = false;

  std::map<std::string, unsigned> loaded_meshes_; // avoid reloading dup meshes

  std::vector<uint8_t> is_emissive;

  CS123SceneGlobalData global_data_; // TODO: what is global data for....
};
} // namespace scenefile_compat
} // namespace scene
