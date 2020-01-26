#pragma once

#include "lib/cuda/utils.h"
#include "scene/color.h"
#include "scene/texture.h"

#include <thrust/optional.h>

namespace scene {
struct Material {
  Color diffuse;
  Color ambient;
  Color reflective;
  Color specular;

#if 0
  Color transparent;
  Color emissive;
#endif

  thrust::optional<TextureData> texture_data;

  float diffuse_blend;
  float ambient_blend;

  float shininess;

  float ior; // index of refraction

  HOST_DEVICE
  Material(const Color &diffuse, const Color &ambient, const Color &reflective,
           const Color &specular, const Color & /*transparent*/,
           const Color & /*emissive*/,
           thrust::optional<TextureData> texture_map_index, float diffuse_blend,
           float ambient_blend, float shininess, float ior)
      : diffuse(diffuse), ambient(ambient), reflective(reflective),
        specular(specular),
#if 0
        transparent(transparent), emissive(emissive),
#endif
        texture_data(texture_map_index), diffuse_blend(diffuse_blend),
        ambient_blend(ambient_blend), shininess(shininess), ior(ior) {
  }

  HOST_DEVICE Material() {}
};
} // namespace scene
