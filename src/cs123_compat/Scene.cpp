#include "cs123_compat/Scene.h"
#include "cs123_compat/CS123ISceneParser.h"

#include <iostream>

namespace Shapes = CS123::Shapes;

namespace CS123 {
Scene::Scene() {}

Scene::~Scene() {
  // Do not delete m_camera, it is owned by SupportCanvas3D
}

void Scene::parse(Scene *sceneToFill, const CS123ISceneParser &parser) {
  sceneToFill->lights_.clear();
  sceneToFill->shape_data_.clear();

  int num_lights = parser.getNumLights();
  for (int i = 0; i < num_lights; i++) {
    CS123SceneLightData v;
    bool out = parser.getLightData(i, v);
    assert(out);
    sceneToFill->addLight(v);
  }

  auto get_transform_matrix = [](const Eigen::Affine3f &mat,
                                 CS123SceneTransformation *transform) {
    switch (transform->type) {
    case TransformationType::Translate:
      return static_cast<Eigen::Affine3f>(
          mat * Eigen::Translation3f(transform->translate));
    case TransformationType::Matrix:
      assert(false);
      return mat;
    case TransformationType::Rotate:
      return static_cast<Eigen::Affine3f>(
          mat * Eigen::AngleAxis(transform->angle, transform->rotate));
    case TransformationType::Scale:
      return static_cast<Eigen::Affine3f>(mat *
                                          Eigen::Scaling(transform->scale));
    default:
      return mat;
    }
  };

  // recursive lambda
  auto parse_node = [&](CS123SceneNode *node) {
    auto parse_node_impl = [&](CS123SceneNode *node,
                               const Eigen::Affine3f &upper_transform,
                               auto &parse_node) -> void {
      Eigen::Affine3f transform_mat = Eigen::Affine3f::Identity();
      for (const auto &transform : node->transformations) {
        transform_mat = get_transform_matrix(transform_mat, transform);
      }

      transform_mat = upper_transform * transform_mat;

      for (const auto &primitive : node->primitives) {
        sceneToFill->addPrimitive(*primitive, transform_mat);
      }

      for (const auto &children : node->children) {
        parse_node(children, transform_mat, parse_node);
      }
    };
    return parse_node_impl(node, Eigen::Affine3f::Identity(), parse_node_impl);
  };

  if (parser.getRootNode() != nullptr) {
    parse_node(parser.getRootNode());
  }

  CS123SceneGlobalData global;
  bool out = parser.getGlobalData(global);
  assert(out);

  for (auto &shape_data : sceneToFill->shape_data_) {
    shape_data.material.cAmbient *= global.ka;
    shape_data.material.cDiffuse *= global.kd;
    shape_data.material.cSpecular *= global.ks;
    shape_data.material.cReflective *= global.ks;
    shape_data.material.cTransparent *= global.kt;
  }
}

scene::TextureImage from_qimage(const QImage &image) {
  scene::TextureImage texture;

  texture.width = static_cast<size_t>(image.width());
  texture.height = static_cast<size_t>(image.height());
  auto bytes = reinterpret_cast<const uint8_t *>(image.bits());
  size_t pixels = static_cast<size_t>(image.byteCount()) / 4;
  texture.data.resize(pixels);
  for (size_t i = 0; i < pixels; i++) {
    texture.data[i] =
        scene::Color(bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2]) / 255.0f;
  }

  return texture;
}

thrust::optional<scene::TextureData>
Scene::getKeyAddTexture(const CS123SceneFileMap &file) {
  if (file.isUsed) {
    auto it = texture_file_name_indexes_.find(file.filename);
    size_t index;
    if (it == texture_file_name_indexes_.end()) {
      index = textures_.size();
      texture_file_name_indexes_.insert(std::make_pair(file.filename, index));
      auto image = QImage(QString::fromStdString(file.filename));
      if (image.isNull()) {
        std::cout
            << "couldn't load texture from file, proceeding without texture"
            << std::endl;
        return thrust::nullopt;
      }
      auto image_converted = image.convertToFormat(QImage::Format_RGB32);
      textures_.push_back(from_qimage(image_converted));
    } else {
      index = it->second;
    }

    return scene::TextureData(index, file.repeatU, file.repeatV);
  } else {
    return thrust::nullopt;
  }
}

void Scene::addPrimitive(const CS123ScenePrimitive &scenePrimitive,
                         const Eigen::Affine3f &matrix) {
  shape_data_.push_back(
      ShapeData(scenePrimitive.type, scenePrimitive.material, matrix,
                getKeyAddTexture(scenePrimitive.material.textureMap)));
}

Scene::ShapeData::ShapeData(PrimitiveType type,
                            const CS123SceneMaterial &material,
                            const Eigen::Affine3f &transform,
                            const thrust::optional<scene::TextureData> &texture)
    : type(type), material(material), transform(transform), texture(texture) {}

void Scene::addLight(const CS123SceneLightData &sceneLight) {
  lights_.push_back(sceneLight);
}
} // namespace CS123
