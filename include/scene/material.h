#pragma once

#include "lib/cuda_utils.h"

#include <Eigen/Core>

#include <thrust/optional.h>

namespace scene {
using Color = Eigen::Array3f;

struct Material {
  Color diffuse;
  Color ambient;
  Color reflective;
  Color specular;
  Color transparent;
  Color emissive;

  thrust::optional<unsigned> texture_map_index; // TODO

  float blend;

  float shininess;

  float ior; // index of refraction

  HOST_DEVICE Material(const Color &diffuse, const Color &ambient,
                       const Color &reflective, const Color &specular,
                       const Color &transparent, const Color &emissive,
                       thrust::optional<unsigned> texture_map_index,
                       float blend, float shininess, float ior)
      : diffuse(diffuse), ambient(ambient), reflective(reflective),
        specular(specular), transparent(transparent), emissive(emissive),
        texture_map_index(texture_map_index), blend(blend),
        shininess(shininess), ior(ior) {}

  HOST_DEVICE Material() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace scene
