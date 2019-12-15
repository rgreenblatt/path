#include "scene/scene.h"

#include "ray/kdtree.h"

namespace scene {
void Scene::copyInTextureRefs() {
  std::transform(textures_.begin(), textures_.end(),
                 std::back_inserter(textures_refs_),
                 [&](const TextureImage &image) { return image.to_ref(); });

  min_bound_ = Eigen::Vector3f(std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max());
  max_bound_ = Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest());
  for (const auto &shape : shapes_) {
    auto [min_bound, max_bound] = ray::detail::get_shape_bounds(shape);
    min_bound_ = min_bound.cwiseMin(min_bound);
    max_bound_ = max_bound.cwiseMax(max_bound);
  }
}
} // namespace scene
