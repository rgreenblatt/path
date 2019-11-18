#pragma once

#include <Eigen/Dense>

namespace scene {
using Color = Eigen::Vector3f;

struct Material {
  Color diffuse;
  Color ambient;
  Color reflective;
  Color specular;
  Color transparent;
  Color emissive;

  int texture_map_index; // TODO

  float blend;

  float shininess;

  float ior; // index of refraction

  Material(const Color &diffuse, const Color &ambient, const Color &reflective,
           const Color &specular, const Color &transparent,
           const Color &emissive, int texture_map_index, float blend,
           float shininess, float ior)
      : diffuse(diffuse), ambient(ambient), reflective(reflective),
        specular(specular), transparent(transparent), emissive(emissive),
        texture_map_index(texture_map_index), blend(blend),
        shininess(shininess), ior(ior) {}
};
} // namespace scene
