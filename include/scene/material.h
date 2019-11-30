#pragma once

#include "lib/cuda_utils.h"
#include "scene/color.h"
#include "scene/texture.h"

#include <thrust/optional.h>

namespace scene {
struct Material {
  Color diffuse;
  Color ambient;
  Color reflective;
  Color specular;
  Color transparent;
  Color emissive;

  thrust::optional<TextureData> texture_data;

  float diffuse_blend;
  float ambient_blend;

  float shininess;

  float ior; // index of refraction

  HOST_DEVICE Material(const Color &diffuse, const Color &ambient,
                       const Color &reflective, const Color &specular,
                       const Color &transparent, const Color &emissive,
                       thrust::optional<TextureData> texture_map_index,
                       float diffuse_blend, float ambient_blend,
                       float shininess, float ior)
      : diffuse(diffuse), ambient(ambient), reflective(reflective),
        specular(specular), transparent(transparent), emissive(emissive),
        texture_data(texture_map_index), diffuse_blend(diffuse_blend),
        ambient_blend(ambient_blend), shininess(shininess), ior(ior) {}

  HOST_DEVICE Material() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace scene
