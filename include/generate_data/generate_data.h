#pragma once

#include <ATen/Tensor.h>

namespace generate_data {
struct PolygonInput {
  at::Tensor point_values;
  at::Tensor overall_features;
  at::Tensor counts;

  // these are all flattened like point_values
  at::Tensor prefix_sum_counts;
  at::Tensor item_to_left_idxs;
  at::Tensor item_to_right_idxs;

  inline PolygonInput to(const at::Tensor &example_tensor) const {
    return {
        .point_values =
            point_values.to(example_tensor.device(), example_tensor.dtype()),
        .overall_features = overall_features.to(example_tensor.device(),
                                                example_tensor.dtype()),
        .counts = counts.to(example_tensor.device()),
        .prefix_sum_counts = prefix_sum_counts.to(example_tensor.device()),
        .item_to_left_idxs = item_to_left_idxs.to(example_tensor.device()),
        .item_to_right_idxs = item_to_right_idxs.to(example_tensor.device()),
    };
  }
};

struct RayInput {
  at::Tensor values;
  at::Tensor counts;
  at::Tensor prefix_sum_counts;
  at::Tensor is_ray;

  inline RayInput to(const at::Tensor &example_tensor) const {
    return {
        .values = values.to(example_tensor.device(), example_tensor.dtype()),
        .counts = counts.to(example_tensor.device()),
        .prefix_sum_counts = prefix_sum_counts.to(example_tensor.device()),
        .is_ray = is_ray.to(example_tensor.device()),
    };
  }
};

struct PolygonInputForTri {
  PolygonInput polygon_feature;
  int tri_idx;

  inline PolygonInputForTri to(const at::Tensor &example_tensor) const {
    return {
        .polygon_feature = polygon_feature.to(example_tensor),
        .tri_idx = tri_idx,
    };
  }
};

} // namespace generate_data
