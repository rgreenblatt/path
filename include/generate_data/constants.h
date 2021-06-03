#pragma once

namespace generate_data {
struct Constants {
  // these don't really need to be accesible?
  int n_multiscale;
  int n_all;

  int n_dims;
  int n_tri_values;
  int n_baryo_dims;
  int n_coords_feature_values;
  int n_poly_point_values;
  int n_rgb_dims;
  int n_shadowable_tris;
  int n_poly_feature_values;
  int n_ray_item_values;
  int n_ray_items;

  Constants();
};

const extern Constants constants;
} // namespace generate_data
