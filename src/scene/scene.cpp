#include "scene/scene.h"
#include "scene/texture_qimage.h"

#include "ray/kdtree.h"
#include <iostream>

namespace scene {
void Scene::finishConstructScene() {
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
    min_bound_ = min_bound_.cwiseMin(min_bound);
    max_bound_ = max_bound_.cwiseMax(max_bound);
  }
}

TextureData Scene::loadTexture(const std::string &file) {
  auto image_tex = load_qimage(file);

  if (!image_tex.has_value()) {
    std::cout << "couldn't load texture from file, exiting" << std::endl;
    std::exit(1);
  }

  return TextureData(addTexture(*image_tex), 1, 1);
}
} // namespace scene
