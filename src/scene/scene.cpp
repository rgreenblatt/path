#include <scene/scene.h>

namespace scene {
unsigned Scene::get_start_shape(Shape shape) const {
  switch (shape) {
  case Shape::Sphere:
    return start_spheres();
  case Shape::Cylinder:
    return start_cylinders();
  case Shape::Cube:
    return start_cubes();
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
  default:
    // invalid shape
    assert(false);
  }
}
} // namespace scene
