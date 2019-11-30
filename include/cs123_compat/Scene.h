#pragma once

#include "cs123_compat/CS123SceneData.h"
#include "scene/material.h"

#include <Eigen/Geometry>
#include <QImage>
#include <thrust/optional.h>

#include <memory>

class Camera;
class CS123ISceneParser;

namespace CS123 {
namespace Shapes {
class Shape;
}
} // namespace CS123

/**
 * @class Scene
 *
 * @brief This is the base class for all scenes. Modify this class if you want
 * to provide common functionality to all your scenes.
 */
namespace CS123 {
class Scene {
  using Shape = CS123::Shapes::Shape;

public:
  Scene();
  virtual ~Scene();

  virtual void settingsChanged() {}

  static void parse(Scene *sceneToFill, const CS123ISceneParser &parser);

  struct ShapeData {
    PrimitiveType type;
    CS123SceneMaterial material;
    Eigen::Affine3f transform;
    thrust::optional<scene::TextureData> texture;

    ShapeData(PrimitiveType type, const CS123SceneMaterial &material,
              const Eigen::Affine3f &transform,
              const thrust::optional<scene::TextureData> &texture);
  };

  // Adds a primitive to the scene.
  virtual void addPrimitive(const CS123ScenePrimitive &scenePrimitive,
                            const Eigen::Affine3f &matrix);

  // Adds a light to the scene.
  virtual void addLight(const CS123SceneLightData &sceneLight);

  thrust::optional<scene::TextureData>
  getKeyAddTexture(const CS123SceneFileMap &file);

  std::vector<ShapeData> shape_data_;
  std::vector<CS123SceneLightData> lights_;
  std::vector<scene::TextureImage> textures_;

private:
  std::map<std::string, size_t> texture_file_name_indexes_;
};
} // namespace CS123
