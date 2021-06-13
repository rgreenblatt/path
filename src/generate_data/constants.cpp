#include "generate_data/single_triangle/constants.h"

#include "generate_data/value_adder.h"

namespace generate_data {
Constants::Constants() {
  n_multiscale = ValueAdder<float>::scales.size();
  n_all = n_multiscale + 1;
  n_dims = 3;
  n_baryo_dims = 2;

  n_poly_point_values = n_baryo_dims + 2 * (1 + n_baryo_dims) + n_all * n_dims +
                        2 * (n_all + n_dims) + 2;

  // centroid (3d and 2d) area and properly scaled area
  n_poly_feature_values = n_dims * n_all + n_baryo_dims + 1 + n_all;

  n_ray_item_values =
      2 * (n_baryo_dims + n_all * n_dims) + n_baryo_dims + 1 + 3 * n_all;
  n_ray_items = 2;

  n_rgb_dims = 3;
}

const Constants constants;
} // namespace generate_data
