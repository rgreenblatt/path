#include "generate_data/single_triangle/constants.h"

#include "generate_data/value_adder.h"

namespace generate_data {
namespace single_triangle {
Constants::Constants() : generate_data::Constants() {
  n_tris = 3;
  n_scene_values = n_tris * (n_all + 2 + 2 * (n_all + n_dims + 4));
  n_polys = n_tris + n_shadowable_tris * 3;
  unsigned points_per_tri = 3;
  n_tri_values =
      n_dims * points_per_tri * n_all + 2 * n_dims * n_all + n_dims + n_all * 2;
  n_coords_feature_values = n_baryo_dims + n_dims * n_all; // baryo and 3d
  n_shadowable_tris = 2;                                   // onto and light
}

const Constants constants;
} // namespace single_triangle
} // namespace generate_data
