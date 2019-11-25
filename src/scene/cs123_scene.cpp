#include "scene/cs123_scene.h"
#include "cs123_compat/CS123ISceneParser.h"
#include "cs123_compat/CS123XmlSceneParser.h"
#include "cs123_compat/Scene.h"

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
    ShapeData next(shape.transform,
                   Material(material.cDiffuse, material.cAmbient,
                            material.cReflective, material.cSpecular,
                            material.cTransparent, material.cEmissive,
                            thrust::nullopt, material.blend, material.shininess,
                            material.ior));

    switch (shape.type) {
    case PrimitiveType::Sphere:
      spheres.push_back(next);
      break;
    case PrimitiveType::Cylinder:
      cylinders.push_back(next);
      break;
    case PrimitiveType::Cube:
      cubes.push_back(next);
      break;
    case PrimitiveType::Cone:
    default:
      cones.push_back(next);
      break;
    }
  }

  num_spheres_ = spheres.size();
  num_cubes_ = cubes.size();
  num_cylinders_ = cylinders.size();
  num_cones_ = cones.size();

  std::copy(spheres.begin(), spheres.end(), std::back_inserter(shapes_));
  std::copy(cylinders.begin(), cylinders.end(), std::back_inserter(shapes_));
  std::copy(cubes.begin(), cubes.end(), std::back_inserter(shapes_));
  std::copy(cones.begin(), cones.end(), std::back_inserter(shapes_));

  for (const auto &light : scene.lights_) {
    switch (light.type) {
    case LightType::Directional:
      lights_.push_back(Light(light.color, DirectionalLight(light.dir)));
      break;
    case LightType::Point:
    default:
      lights_.push_back(
          Light(light.color, PointLight(light.pos, light.function)));
      break;
    }
  }

  CS123SceneCameraData camera_data;

  parser.getCameraData(camera_data);

  auto w = -camera_data.look.normalized().eval();
  auto v = (camera_data.up - camera_data.up.dot(w) * w).normalized().eval();
  auto u = v.cross(w).normalized();

  Eigen::Matrix3f mat;
  

  mat.row(0) = u;
  mat.row(1) = v;
  mat.row(2) = w;

  float theta_h = camera_data.heightAngle * M_PI / 180.0f;
  float theta_w = std::atan(width * std::tan(theta_h / 2) / height) * 2;
  float far = 30.0f;

  transform_ =
      (mat * Eigen::Translation3f(-camera_data.pos)).inverse() *
      Eigen::Scaling(Eigen::Vector3f(1.0f / (std::tan(theta_w / 2) * far),
                                     1.0f / (std::tan(theta_h / 2) * far),
                                     1.0f / far))
          .inverse();
}
} // namespace scene
