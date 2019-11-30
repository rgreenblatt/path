#include <scene/scene.h>

#include <dbg.h>

namespace scene {
void Scene::copy_in_texture_refs() {
  std::transform(textures_.begin(), textures_.end(),
                 std::back_inserter(textures_refs_),
                 [&](const TextureImage &image) { return image.to_ref(); });

}

unsigned Scene::get_start_shape(Shape shape) const {
  switch (shape) {
  case Shape::Sphere:
    return start_spheres();
  case Shape::Cylinder:
    return start_cylinders();
  case Shape::Cube:
    return start_cubes();
  case Shape::Cone:
    return start_cones();
  default:
    // invalid shape
    assert(false);
  }
}
unsigned Scene::get_num_shape(Shape shape) const {
  switch (shape) {
  case Shape::Sphere:
    return num_spheres();
  case Shape::Cylinder:
    return num_cylinders();
  case Shape::Cube:
    return num_cubes();
  case Shape::Cone:
    return num_cones();
  default:
    // invalid shape
    assert(false);
  }
}
} // namespace scene
