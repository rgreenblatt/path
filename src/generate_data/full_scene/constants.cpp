#include "generate_data/full_scene/constants.h"

#include "generate_data/value_adder.h"

namespace generate_data {
namespace full_scene {
Constants::Constants() : generate_data::Constants() {
  unsigned points_per_tri = 3;
  n_tri_values =
      n_dims * points_per_tri * n_all + n_dims * n_all + n_dims + n_all;
  n_bsdf_values = n_rgb_dims * 4 + 2;
  n_coords_feature_values = n_baryo_dims + n_dims; // baryo and 3d
}

const Constants constants;
} // namespace full_scene
} // namespace generate_data
