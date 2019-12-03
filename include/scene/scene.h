#pragma once

#include "lib/unified_memory_vector.h"
#include "scene/light.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

#include <vector>

namespace scene {
class Scene {
public:
  unsigned numSpheres() const { return num_spheres_; }
  unsigned numCylinders() const { return num_cylinders_; }
  unsigned numCubes() const { return num_cubes_; }
  unsigned numCones() const { return num_cones_; }
  unsigned numShapes() const { return shapes_.size(); }

  unsigned startSpheres() const { return 0; }
  unsigned startCylinders() const { return num_spheres_; }
  unsigned startCubes() const { return num_spheres_ + num_cylinders_; }
  unsigned startCones() const {
    return num_spheres_ + num_cylinders_ + num_cubes_;
  }

  virtual const ShapeData *getShapes() const { return shapes_.data(); }

  virtual const Light *getLights() const { return lights_.data(); }
  virtual unsigned getNumLights() const { return lights_.size(); }

  unsigned getNumShape(Shape shape) const;
  unsigned getStartShape(Shape shape) const;

  unsigned getNumTextures() const { return textures_refs_.size(); }
  const TextureImageRef* getTextures() const { return textures_refs_.data(); }

protected:
  std::vector<ShapeData> shapes_;
  ManangedMemVec<Light> lights_;
  ManangedMemVec<TextureImage> textures_;

  unsigned num_spheres_;
  unsigned num_cylinders_;
  unsigned num_cubes_;
  unsigned num_cones_;
  
  void copyInTextureRefs();

private:
  ManangedMemVec<TextureImageRef> textures_refs_;
};
} // namespace scene
