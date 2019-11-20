#include "scene/cs123_scene.h"
#include "cs123_compat/CS123ISceneParser.h"
#include "cs123_compat/CS123XmlSceneParser.h"
#include "cs123_compat/Scene.h"

namespace scene {
CS123Scene::CS123Scene(const std::string &file_name) {

  num_spheres_ = 0;
  num_cylinders_ = 0;
  num_cubes_ = 0;

  CS123XmlSceneParser parser(file_name);
  if (!parser.parse()) {
    return;
  }

  CS123::Scene scene;
  CS123::Scene::parse(&scene, parser);

  std::vector<ShapeData> spheres;
  std::vector<ShapeData> cylinders;
  std::vector<ShapeData> cubes;

  for (const auto &shape : scene.shape_data_) {
    const auto &material = shape.material;
    ShapeData next(shape.transform,
                   Material(material.cDiffuse, material.cAmbient,
                            material.cReflective, material.cSpecular,
                            material.cTransparent, material.cEmissive,
                            thrust::nullopt, material.blend, material.shininess,
                            material.ior));

    switch (shape.type) {
    case PrimitiveType::Cube:
      cubes.push_back(next);
      break;
    case PrimitiveType::Sphere:
      spheres.push_back(next);
      break;
    case PrimitiveType::Cylinder:
    default:
      cylinders.push_back(next);
      break;
    }
  }

  std::copy(spheres.begin(), spheres.end(), std::back_inserter(shapes_));
  std::copy(cylinders.begin(), cylinders.end(), std::back_inserter(shapes_));
  std::copy(cubes.begin(), cubes.end(), std::back_inserter(shapes_));

  num_spheres_ = spheres.size();
  num_cubes_ = cubes.size();
  num_cylinders_ = cylinders.size();


  for (const auto &light : scene.lights_) {
      switch (light.type) {
      case LightType::Directional:
        lights_.push_back(Light(light.color, DirectionalLight(light.dir)));
      case LightType::Point:
      default:
        lights_.push_back(
            Light(light.color, PointLight(light.pos, light.function)));
      }
    }
}
} // namespace scene
