#pragma once

#include "lib/unified_memory_vector.h"
#include "scene/scene.h"

#include <string>

namespace scene {
class CS123Scene : public Scene {
public:
  CS123Scene(const std::string &file_path);
  unsigned num_spheres() const override { return num_spheres_; }
  unsigned num_cylinders() const override { return num_cylinders_; }
  unsigned num_cubes() const override { return num_cubes_; }

  unsigned start_spheres() const override { return 0; }
  unsigned start_cylinders() const override { return num_spheres_; }
  unsigned start_cubes() const override {
    return num_spheres_ + num_cylinders_;
  }

  const ShapeData *get_shapes() const override { return shapes_.data(); }

  const Light *get_lights() const override { return lights_.data(); }
  unsigned get_num_lights() const override { return lights_.size(); }

protected:
  ManangedMemVec<ShapeData> shapes_;
  ManangedMemVec<Light> lights_;

  unsigned num_spheres_;
  unsigned num_cylinders_;
  unsigned num_cubes_;
};
} // namespace scene
