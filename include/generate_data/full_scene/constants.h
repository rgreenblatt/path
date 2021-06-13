#pragma once

#include "generate_data/constants.h"

namespace generate_data {
namespace full_scene {
struct Constants : generate_data::Constants {
  int n_tri_values;
  int n_coords_feature_values;
  int n_bsdf_values;

  Constants();
};

const extern Constants constants;
} // namespace full_scene
} // namespace generate_data
