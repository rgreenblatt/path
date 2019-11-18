#include "scene/pool_scene.h"

namespace scene {
PoolScene::PoolScene() {
  shapes_.push_back(ShapeData(
      Transform::Identity(), Material(Color(1, 0, 0), Color(), Color(), Color(),
                                      Color(), Color(), -1, 0, 0, 0)));

  num_spheres_ = 1;
  num_cylinders_ = 0;
  num_cubes_ = 0;
}
} // namespace scene
