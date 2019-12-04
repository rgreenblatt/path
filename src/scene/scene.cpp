#include <scene/scene.h>

#include <dbg.h>

namespace scene {
void Scene::copyInTextureRefs() {
  std::transform(textures_.begin(), textures_.end(),
                 std::back_inserter(textures_refs_),
                 [&](const TextureImage &image) { return image.to_ref(); });
}

uint16_t Scene::getStartShape(Shape shape) const {
  switch (shape) {
  case Shape::Sphere:
    return startSpheres();
  case Shape::Cylinder:
    return startCylinders();
  case Shape::Cube:
    return startCubes();
  case Shape::Cone:
    return startCones();
  default:
    // invalid shape
    assert(false);
  }
}
uint16_t Scene::getNumShape(Shape shape) const {
  switch (shape) {
  case Shape::Sphere:
    return numSpheres();
  case Shape::Cylinder:
    return numCylinders();
  case Shape::Cube:
    return numCubes();
  case Shape::Cone:
    return numCones();
  default:
    // invalid shape
    assert(false);
  }
}
} // namespace scene
