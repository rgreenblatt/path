#pragma once

#include "intersect/triangle.h"
#include "lib/unit_vector.h"
#include "lib/vector_type.h"
#include "rng/rng.h"
#include "scene/scene.h"

#include <array>
#include <random>

namespace generate_data {
struct TriangleNormals {
  intersect::Triangle tri;
  std::array<UnitVector, 3> normals;
};

class MeshSceneGenerator {
public:
  MeshSceneGenerator();

  const scene::Scene &generate(std::mt19937 &rng);

private:
  void add_mesh(const VectorT<TriangleNormals> &tris,
                const Eigen::Affine3f &transform, unsigned material_idx);

  scene::Scene scene_;
  intersect::accel::AABB overall_aabb_;
  VectorT<TriangleNormals> sphere_;
  VectorT<TriangleNormals> monkey_;
  VectorT<TriangleNormals> torus_;
  VectorT<const VectorT<TriangleNormals> *> meshs_;
};
} // namespace generate_data
