#include "scene/pool_scene.h"

namespace scene {
PoolScene::PoolScene() {
  shapes_.push_back(ShapeData(
      Transform::Identity(), Material(Color(1, 0, 0), Color(), Color(), Color(),
                                      Color(), Color(), -1, 0, 0, 0)));
  /* shapes_.push_back( */
  /*     ShapeData(static_cast<Transform>(Eigen::Translation3f(-3, 2, 1)), */
  /*               Material(Color(1, 0, 0), Color(), Color(), Color(), Color(), */
  /*                        Color(), -1, 0, 0, 0))); */
  /* shapes_.push_back( */
  /*     ShapeData(static_cast<Transform>(Eigen::Translation3f(3, 4, 1)), */
  /*               Material(Color(1, 0, 0), Color(), Color(), Color(), Color(), */
  /*                        Color(), -1, 0, 0, 0))); */

  num_spheres_ = 0;
  num_cylinders_ = 1;
  num_cubes_ = 0;
}
} // namespace scene
