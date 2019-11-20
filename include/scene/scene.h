#pragma once

#include <vector>

#include <scene/light.h>
#include <scene/shape.h>
#include <scene/shape_data.h>

namespace scene {
class Scene {
public:
  virtual unsigned num_spheres() const = 0;
  virtual unsigned num_cylinders() const = 0;
  virtual unsigned num_cubes() const = 0;

  virtual unsigned start_spheres() const = 0;
  virtual unsigned start_cylinders() const = 0;
  virtual unsigned start_cubes() const = 0;

  virtual const ShapeData *get_shapes() const = 0;

  virtual const Light *get_lights() const = 0;
  virtual unsigned get_num_lights() const = 0;

  unsigned get_num_shape(Shape shape) const;
  unsigned get_start_shape(Shape shape) const;
};
} // namespace scene
