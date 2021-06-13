#pragma once

#include "lib/vector_type.h"
#include "scene/scene.h"

#include <tuple>

namespace generate_data {
namespace full_scene {
struct IntersectedBaryocentricCoords {
  VectorT<std::tuple<unsigned, unsigned>> image_indexes;
  VectorT<std::tuple<float, float>> coords;
  VectorT<UnitVector> directions;
  VectorT<unsigned> tri_idxs;
};

IntersectedBaryocentricCoords
intersect_for_baryocentric_coords(const scene::Scene &scene, unsigned dim);
} // namespace full_scene
} // namespace generate_data
