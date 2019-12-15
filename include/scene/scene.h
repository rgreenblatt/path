#pragma once

#include "lib/unified_memory_vector.h"
#include "scene/light.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

#include <set>
#include <vector>

namespace scene {
class Scene {
public:
  const ShapeData *getShapes() const { return shapes_.data(); }
  uint16_t getNumShapes() const { return shapes_.size(); }

  const Light *getLights() const { return lights_.data(); }
  unsigned getNumLights() const { return lights_.size(); }

  uint16_t getNumTextures() const { return textures_refs_.size(); }
  const TextureImageRef *getTextures() const { return textures_refs_.data(); }

  Eigen::Vector3f getMinBound() const { return min_bound_; }

  Eigen::Vector3f getMaxBound() const { return max_bound_; };

  const std::set<uint16_t> &updatedShapes() const { return updated_shapes_; }

  void clearUpdates() { updated_shapes_.clear(); }

  virtual void step(float secs) = 0;

  virtual ~Scene() {}

protected:
  void copyInTextureRefs();

  unsigned addShape(const ShapeData &shape_data) {
    unsigned index = shapes_.size();
    shapes_.push_back(shape_data);
    return index;
  };

  void addLight(const Light &light) { lights_.push_back(light); }

  unsigned addTexture(const TextureImage &tex) {
    unsigned index = textures_.size();
    textures_.push_back(tex);

    return index;
  }

  void updateTransformShape(uint16_t shape_index,
                            const Eigen::Affine3f &new_transform) {
    updated_shapes_.insert(shape_index);
    shapes_[shape_index].set_transform(new_transform);
  }

  Eigen::Vector3f min_bound_;
  Eigen::Vector3f max_bound_;

private:
  std::vector<ShapeData> shapes_;
  std::set<uint16_t> updated_shapes_;
  ManangedMemVec<Light> lights_;
  ManangedMemVec<TextureImage> textures_;

  ManangedMemVec<TextureImageRef> textures_refs_;
};
} // namespace scene
