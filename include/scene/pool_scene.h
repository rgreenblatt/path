#pragma once

#include "lib/unified_memory_vector.h"
#include "scene/scene.h"

namespace scene {
class PoolScene : Scene {
public:
  PoolScene();
  const ShapeData *spheres() const override { return shapes_.data(); }
  const ShapeData *cylinders() const override { return &shapes_[num_spheres_]; }
  const ShapeData *cubes() const override {
    return &shapes_[num_spheres_ + num_cylinders_];
  }
  int num_spheres() const override { return num_spheres_; }
  int num_cylinders() const override { return num_cylinders_; }
  int num_cubes() const override { return num_cubes_; }

protected:
  AllocVec<ShapeData> shapes_;

  int num_spheres_;
  int num_cylinders_;
  int num_cubes_;
};
} // namespace scene
