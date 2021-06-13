#pragma once

#include "generate_data/generate_data.h"

#include <ATen/Tensor.h>

#include <tuple>

namespace generate_data {
namespace single_triangle {
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
  at::Tensor is_close_point;

  inline RayInput to(const at::Tensor &example_tensor) const {
    return {
        .values = values.to(example_tensor.device(), example_tensor.dtype()),
        .counts = counts.to(example_tensor.device()),
        .prefix_sum_counts = prefix_sum_counts.to(example_tensor.device()),
        .is_ray = is_ray.to(example_tensor.device()),
        .is_close_point = is_close_point.to(example_tensor.device()),
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

struct NetworkInputs {
  at::Tensor overall_scene_features;
  at::Tensor triangle_features;
  std::vector<PolygonInputForTri> polygon_inputs;
  std::vector<RayInput> ray_inputs;
  at::Tensor baryocentric_coords;

  inline NetworkInputs to(const at::Tensor &example_tensor) const {
    std::vector<PolygonInputForTri> new_polygon_inputs(polygon_inputs.size());
    std::transform(
        polygon_inputs.begin(), polygon_inputs.end(),
        new_polygon_inputs.begin(),
        [&](const PolygonInputForTri &inp) { return inp.to(example_tensor); });
    std::vector<RayInput> new_ray_inputs(ray_inputs.size());
    std::transform(ray_inputs.begin(), ray_inputs.end(), new_ray_inputs.begin(),
                   [&](const RayInput &inp) { return inp.to(example_tensor); });
    return {
        .overall_scene_features = overall_scene_features.to(
            example_tensor.device(), example_tensor.dtype()),
        .triangle_features = triangle_features.to(example_tensor.device(),
                                                  example_tensor.dtype()),
        .polygon_inputs = new_polygon_inputs,
        .ray_inputs = new_ray_inputs,
        .baryocentric_coords = baryocentric_coords.to(example_tensor.device(),
                                                      example_tensor.dtype()),
    };
  }
};

StandardData<NetworkInputs> generate_data(int n_scenes, int n_samples_per_scene,
                                          int n_samples, unsigned base_seed);

ImageData<NetworkInputs> generate_data_for_image(int n_scenes, int dim,
                                                 int n_samples,
                                                 unsigned base_seed);

// hack to avoid issues with "CUDA free failed: cudaErrorCudartUnloading:
// driver shutting down"
void deinit_renderers();
} // namespace single_triangle
} // namespace generate_data
