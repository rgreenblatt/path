#include <scene/scene.h>

#include <iostream>

namespace scene {
const ShapeData *Scene::get_shapes(Shape shape) const {
  switch (shape) {
  case Shape::Sphere:
    return spheres();
  case Shape::Cylinder:
    return cylinders();
  case Shape::Cube:
    return cubes();
  default:
    std::cout << "invalid Shape type" << std::endl;
    assert(false);
  }
}
int Scene::get_num_shapes(Shape shape) const {
  switch (shape) {
  case Shape::Sphere:
    return num_spheres();
  case Shape::Cylinder:
    return num_cylinders();
  case Shape::Cube:
    return num_cubes();
  default:
    std::cout << "invalid Shape type" << std::endl;
    assert(false);
  }
}
} // namespace scene
