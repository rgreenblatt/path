#pragma once

#include "lib/unified_memory_vector.h"
#include "scene/light.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

#include <vector>

namespace scene {
class Scene {
public:
  unsigned num_spheres() const { return num_spheres_; }
  unsigned num_cylinders() const { return num_cylinders_; }
  unsigned num_cubes() const { return num_cubes_; }
  unsigned num_cones() const { return num_cones_; }
  unsigned num_shapes() const { return shapes_.size(); }

  unsigned start_spheres() const { return 0; }
  unsigned start_cylinders() const { return num_spheres_; }
  unsigned start_cubes() const { return num_spheres_ + num_cylinders_; }
  unsigned start_cones() const {
    return num_spheres_ + num_cylinders_ + num_cubes_;
  }

  virtual const ShapeData *get_shapes() const { return shapes_.data(); }

  virtual const Light *get_lights() const { return lights_.data(); }
  virtual unsigned get_num_lights() const { return lights_.size(); }

  unsigned get_num_shape(Shape shape) const;
  unsigned get_start_shape(Shape shape) const;

  unsigned get_num_textures() const { return textures_refs_.size(); }
  const TextureImageRef* get_textures() const { return textures_refs_.data(); }

protected:
  std::vector<ShapeData> shapes_;
  ManangedMemVec<Light> lights_;
  ManangedMemVec<TextureImage> textures_;

  unsigned num_spheres_;
  unsigned num_cylinders_;
  unsigned num_cubes_;
  unsigned num_cones_;
  
  void copy_in_texture_refs();

private:
  ManangedMemVec<TextureImageRef> textures_refs_;
};
} // namespace scene
