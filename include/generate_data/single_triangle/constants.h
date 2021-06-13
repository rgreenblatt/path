#pragma once

#include "generate_data/constants.h"

namespace generate_data {
namespace single_triangle {
struct Constants : generate_data::Constants {
  int n_tris;
  int n_scene_values;
  int n_polys;
  int n_tri_values;
  int n_coords_feature_values;
  int n_shadowable_tris;

  Constants();
};

const extern Constants constants;
} // namespace single_triangle
} // namespace generate_data
