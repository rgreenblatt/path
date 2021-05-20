#pragma once

#include "intersect/triangle.h"
#include "lib/unit_vector.h"
#include "lib/vector_type.h"
#include "rng/rng.h"

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

  // TODO ret type
  void generate(std::mt19937 &rng);

private:
  VectorT<TriangleNormals> sphere_;
  VectorT<TriangleNormals> monkey_;
  VectorT<TriangleNormals> torus_;
};
} // namespace generate_data
