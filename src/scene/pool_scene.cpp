#include "scene/pool_scene.h"

namespace scene {
PoolScene::PoolScene() {
  shapes_.push_back(ShapeData(
      Transform::Identity(),
      Material(Color(0, 1, 0), Color(0.5, 0, 0), Color::Zero(), Color::Zero(),
               Color::Zero(), Color::Zero(), thrust::nullopt, 0, 0, 0, 0)));
  shapes_.push_back(
      ShapeData(static_cast<Transform>(Eigen::Translation3f(-3, 2, 1)),
                Material(Color(0, 0, 1), Color::Zero(), Color(0.3, 0.3, 0.3),
                         Color::Zero(), Color::Zero(), Color::Zero(),
                         thrust::nullopt, 0, 7, 0, 0)));
  shapes_.push_back(ShapeData(
      static_cast<Transform>(Eigen::Translation3f(-4, 3, 2) *
                             Eigen::Scaling(1.0f, 3.0f, 0.5f)),
      Material(Color(0, 0, 1), Color::Zero(), Color::Zero(), Color(0.7, 0, 0),
               Color::Zero(), Color::Zero(), thrust::nullopt, 0, 10, 0, 0)));
  shapes_.push_back(ShapeData(
      static_cast<Transform>(Eigen::Translation3f(-1, 2, 1)),
      Material(Color(0, 0, 1), Color::Zero(), Color::Zero(), Color::Zero(),
               Color::Zero(), Color::Zero(), thrust::nullopt, 0, 0, 0, 0)));
  shapes_.push_back(ShapeData(
      static_cast<Transform>(Eigen::Translation3f(-1, 3, 0)),
      Material(Color(0, 0, 1), Color::Zero(), Color::Zero(), Color::Zero(),
               Color::Zero(), Color::Zero(), thrust::nullopt, 0, 0, 0, 0)));
  shapes_.push_back(ShapeData(
      static_cast<Transform>(Eigen::Translation3f(4, 4, 1)),
      Material(Color(1, 0, 0), Color::Zero(), Color::Zero(), Color::Zero(),
               Color::Zero(), Color::Zero(), thrust::nullopt, 0, 0, 0, 0)));

  num_spheres_ = 3;
  num_cylinders_ = 2;
  num_cubes_ = 1;
  num_cones_ = 1;

  lights_.push_back(
      Light(Color(0.5, 0.5, 0.5),
            PointLight(Eigen::Vector3f(3, 3, 3), Eigen::Array3f(1, 0, 0))));

  copy_in_texture_refs();
}
} // namespace scene
