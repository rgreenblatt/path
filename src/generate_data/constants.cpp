#include "generate_data/constants.h"

#include "generate_data/value_adder.h"

namespace generate_data {
Constants::Constants() {
  unsigned n_multiscale = ValueAdder<float>::scales.size();
  unsigned n_all = n_multiscale + 1;
  n_tris = 3;
  n_dims = 3;
  n_baryo_dims = 2;

  n_scene_values = n_tris * (n_all + 2 + 2 * (n_all + n_dims + 4));

  unsigned points_per_tri = 3;
  n_tri_values =
      n_dims * points_per_tri * n_all + 3 * n_dims * n_all + n_dims + n_all * 2;

  n_coords_feature_values = n_baryo_dims + n_dims * n_all; // baryo and 3d

  n_poly_point_values = n_baryo_dims + 2 * (1 + n_baryo_dims) + n_all * n_dims +
                        2 * (n_all + n_dims) + 2;

  // centroid (3d and 2d) area and properly scaled area
  n_poly_feature_values = n_dims * n_all + n_baryo_dims + 1 + n_all;

  n_shadowable_tris = 2; // onto and light
  n_polys = n_tris + n_shadowable_tris * 3;

  n_ray_item_values =
      2 * (n_baryo_dims + n_all * n_dims) + n_baryo_dims + 1 + 3 * n_multiscale;
  n_ray_items = 2;

  n_rgb_dims = 3;
}

const Constants constants;
} // namespace generate_data
