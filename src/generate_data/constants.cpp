#include "generate_data/constants.h"

namespace generate_data {
Constants::Constants() {
  n_tris = 3;
  n_scene_values = (3 + 2 * 11) * n_tris;
  n_dims = 3;
  n_tri_values = 3 * n_dims + 4 * n_dims + 2;
  n_baryo_dims = 2;
  n_coords_feature_values = n_baryo_dims + n_dims; // baryo and 3d
  n_poly_point_values = 3 * n_baryo_dims + n_dims + 4;
  n_rgb_dims = 3;
  n_shadowable_tris = 2; // onto and light

  // centroid (3d and 2d) area and properly scaled area
  n_poly_feature_values = 3 + 2 + 2;

  n_polys = n_tris + n_shadowable_tris * 3;
}

const Constants constants;
} // namespace generate_data
