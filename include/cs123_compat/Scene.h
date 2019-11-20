#pragma once

#include "cs123_compat/CS123SceneData.h"

#include <Eigen/Geometry>
#include <QImage>
#include <boost/optional.hpp>

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

  struct TextureImage {
    size_t width;
    size_t height;
    std::vector<scene::Color> data;

    TextureImage(const QImage &image);

    inline const scene::Color &index(size_t x, size_t y) const {
      return data[x + y * width];
    }
  };

  struct TextureData {
    size_t index;
    float repeat_u;
    float repeat_v;

    TextureData(size_t index, float repeat_u, float repeat_v);

    // inline
    inline scene::Color sample(const std::vector<TextureImage> &textures,
                               const Eigen::Array2f &uv) const {
      const auto &texture = textures[index];
      size_t x = static_cast<size_t>(uv[0] * static_cast<float>(texture.width) *
                                     repeat_u) %
                 (texture.width - 1);
      size_t y = static_cast<size_t>(
                     uv[1] * static_cast<float>(texture.height) * repeat_v) %
                 (texture.height - 1);

      return texture.index(x, y);
    }
  };

  struct ShapeData {
    PrimitiveType type;
    CS123SceneMaterial material;
    Eigen::Affine3f transform;
    boost::optional<TextureData> texture;

    ShapeData(PrimitiveType type, const CS123SceneMaterial &material,
              const Eigen::Affine3f &transform,
              const boost::optional<TextureData> &texture);
  };

  // Adds a primitive to the scene.
  virtual void addPrimitive(const CS123ScenePrimitive &scenePrimitive,
                            const Eigen::Affine3f &matrix);

  // Adds a light to the scene.
  virtual void addLight(const CS123SceneLightData &sceneLight);

  boost::optional<TextureData> getKeyAddTexture(const CS123SceneFileMap &file);

  std::vector<ShapeData> shape_data_;
  std::vector<CS123SceneLightData> lights_;
  std::vector<TextureImage> textures_;

private:
  std::map<std::string, size_t> texture_file_name_indexes_;
};
} // namespace CS123
