#pragma once

#include <vector>

#include <scene/shape.h>
#include <scene/shape_data.h>

namespace scene {
class Scene {
public:
  virtual const ShapeData *spheres() const = 0;
  virtual const ShapeData *cylinders() const = 0;
  virtual const ShapeData *cubes() const = 0;
  virtual unsigned num_spheres() const = 0;
  virtual unsigned num_cylinders() const = 0;
  virtual unsigned num_cubes() const = 0;

  const ShapeData *get_shapes(Shape shape) const;
  unsigned get_num_shapes(Shape shape) const;
};
} // namespace scene
