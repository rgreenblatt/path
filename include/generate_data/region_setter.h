#pragma once

#include "generate_data/gen_data.h"
#include "generate_data/torch_utils.h"
#include "generate_data/triangle.h"
#include "generate_data/triangle_subset.h"
#include "lib/vector_type.h"

namespace generate_data {
// TODO: consider more efficient representation later
template <unsigned n_prior_dims> struct RegionSetter {
  RegionSetter() = default;

  RegionSetter(const std::array<TorchIdxT, n_prior_dims> &prior_dims);

  double set_region(const std::array<TorchIdxT, n_prior_dims> &prior_idxs_in,
                    const TriangleSubset &region, const Triangle &tri);

  // NOTE: moves values out (can't be used multiple times!)
  ATTR_NO_DISCARD_PURE PolygonInput as_poly_input();

  VectorT<VectorT<float>> point_values;
  std::array<unsigned, n_prior_dims> index_multipliers;
  at::Tensor overall_features;
  at::Tensor counts;
};
} // namespace generate_data
