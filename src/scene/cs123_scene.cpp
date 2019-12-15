#include "scene/cs123_scene.h"
#include "cs123_compat/CS123ISceneParser.h"
#include "cs123_compat/CS123XmlSceneParser.h"
#include "cs123_compat/Scene.h"
#include "scene/camera.h"

#include <iostream>

#include <dbg.h>

namespace scene {

CS123Scene::CS123Scene(const std::string &file_path, unsigned width,
                       unsigned height) {
  CS123XmlSceneParser parser(file_path);
  if (!parser.parse()) {
    return;
  }

  CS123::Scene scene;
  CS123::Scene::parse(&scene, parser);

  std::vector<ShapeData> spheres;
  std::vector<ShapeData> cylinders;
  std::vector<ShapeData> cubes;
  std::vector<ShapeData> cones;

  for (const auto &shape : scene.shape_data_) {
    const auto &material = shape.material;
    auto get_next = [&](scene::Shape shape_type) {
      return ShapeData(shape.transform,
                       Material(material.cDiffuse, material.cAmbient,
                                material.cReflective, material.cSpecular,
                                material.cTransparent, material.cEmissive,
                                shape.texture, material.blend, material.blend,
                                material.shininess, material.ior),
                       shape_type);
    };

    switch (shape.type) {
    case PrimitiveType::Sphere:
      addShape(get_next(scene::Shape::Sphere));
      break;
    case PrimitiveType::Cylinder:
      addShape(get_next(scene::Shape::Cylinder));
      break;
    case PrimitiveType::Cube:
      addShape(get_next(scene::Shape::Cube));
      break;
    case PrimitiveType::Cone:
    default:
      addShape(get_next(scene::Shape::Cone));
      break;
    }
  }

  for (const auto &tex : scene.textures_) {
    addTexture(tex);
  }

  for (const auto &light : scene.lights_) {
    switch (light.type) {
    case LightType::Directional:
      addLight(Light(light.color, DirectionalLight(light.dir)));
      break;
    case LightType::Point:
    default:
      addLight(Light(light.color, PointLight(light.pos, light.function)));
      break;
    }
  }

  CS123SceneCameraData camera_data;

  parser.getCameraData(camera_data);

  auto [film_to_world, world_to_film, unhinging] = get_camera_transform(
      camera_data.look, camera_data.up, camera_data.pos,
      camera_data.heightAngle * M_PI / 180.0f, width, height, 30.0f);

  film_to_world_ = film_to_world;
  world_to_film_ = world_to_film;
  unhinging_ = unhinging;

  finishConstructScene();
}
} // namespace scene
