#pragma once

#include "lib/unified_memory_vector.h"
#include "scene/light.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

#include <vector>

namespace scene {
class Scene {
public:
  uint16_t numSpheres() const { return num_spheres_; }
  uint16_t numCylinders() const { return num_cylinders_; }
  uint16_t numCubes() const { return num_cubes_; }
  uint16_t numCones() const { return num_cones_; }
  uint16_t numShapes() const { return shapes_.size(); }

  uint16_t startSpheres() const { return 0; }
  uint16_t startCylinders() const { return num_spheres_; }
  uint16_t startCubes() const { return num_spheres_ + num_cylinders_; }
  uint16_t startCones() const {
    return num_spheres_ + num_cylinders_ + num_cubes_;
  }

  virtual const ShapeData *getShapes() const { return shapes_.data(); }
  virtual uint16_t getNumShapes() const { return shapes_.size(); }

  virtual const Light *getLights() const { return lights_.data(); }
  virtual unsigned getNumLights() const { return lights_.size(); }

  uint16_t getNumShape(Shape shape) const;
  uint16_t getStartShape(Shape shape) const;

  uint16_t getNumTextures() const { return textures_refs_.size(); }
  const TextureImageRef* getTextures() const { return textures_refs_.data(); }

protected:
  std::vector<ShapeData> shapes_;
  ManangedMemVec<Light> lights_;
  ManangedMemVec<TextureImage> textures_;

  uint16_t num_spheres_;
  uint16_t num_cylinders_;
  uint16_t num_cubes_;
  uint16_t num_cones_;
  
  void copyInTextureRefs();

private:
  ManangedMemVec<TextureImageRef> textures_refs_;
};
} // namespace scene
