#pragma once

namespace generate_data {
struct Constants {
  int n_tris;
  int n_scene_values;
  int n_dims;
  int n_tri_values;
  int n_baryo_dims;
  int n_coords_feature_values;
  int n_poly_point_values;
  int n_rgb_dims;
  int n_shadowable_tris;
  int n_poly_feature_values;
  int n_polys;
  int n_ray_item_values;
  int n_ray_items;

  Constants();
};

const extern Constants constants;
} // namespace generate_data
