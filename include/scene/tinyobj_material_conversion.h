#pragma once

#include "bsdf/material.h"
#include "bsdf/union_bsdf.h"
#include "scene/material.h"

#include <tiny_obj_loader.h>

#include <iostream>

namespace scene {
Material tinyobj_material_conversion(const tinyobj::material_t &material) {
  using bsdf::BSDFType;
  using Union = bsdf::UnionBSDF::Union;

  auto to_eigen = [](const tinyobj::real_t *reals) {
    return Eigen::Array3f(reals[0], reals[1], reals[2]);
  };

  auto diffuse = to_eigen(material.diffuse);
  auto specular = to_eigen(material.specular);
  auto emission = to_eigen(material.emission);
  float shininess = material.shininess;
  float ior = material.ior;
  int illum = material.illum;

  if (illum != 2 && illum != 5) {
    std::cerr << "unhandled illum value: " << illum << std::endl;
    std::cerr << "only 2 and 5 supported" << std::endl;
    abort();
  }

  constexpr float epsilon = 1e-8; // for checking if component is zero;

  bool diffuse_non_zero = diffuse.matrix().squaredNorm() > epsilon;
  bool specular_non_zero = specular.matrix().squaredNorm() > epsilon;

  // above this considered perfect mirror
  const float shininess_threshold = 100;

  if (illum == 5) {
    /* if (diffuse_non_zero || !specular_non_zero) { */
    /*   std::cerr */
    /*       << "diffuse values non-zero or specular values zero for dielectric
     * " */
    /*          "refractive (unhandled)" */
    /*       << std::endl; */
    /*   abort(); */
    /* } */

    // refractive
    return Material{{Union::create<BSDFType::DielectricRefractive>(
                        Eigen::Vector3f::Ones(), ior)},
                    emission};
  } else if (diffuse_non_zero && !specular_non_zero) {
    // ideal diffuse
    return Material{{Union::create<BSDFType::Diffuse>({diffuse})}, emission};
  } else if (shininess > shininess_threshold) {
    if (diffuse_non_zero || !specular_non_zero) {
      std::cerr << "diffuse values non-zero or specular values zero for mirror "
                   "(unhandled)"
                << std::endl;
      abort();
    }

    // ideal specular
    return Material{{Union::create<BSDFType::Mirror>({specular})}, emission};
  } else if (specular_non_zero /*&& !diffuse_non_zero*/) {
    return Material{{Union::create<BSDFType::Glossy>(specular, shininess)},
                    emission};
  } else {
    // TODO
    std::cerr << "unhandled material settings" << std::endl;
    abort();
  }
}
} // namespace scene