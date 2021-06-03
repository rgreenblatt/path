#pragma once

#include "generate_data/generate_data.h"

#include <ATen/Tensor.h>

#include <tuple>

namespace generate_data {
namespace single_triangle {
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

struct StandardData {
  NetworkInputs inputs;
  at::Tensor values;

  inline StandardData to(const at::Tensor &example_tensor) const {
    return {
        .inputs = inputs.to(example_tensor),
        .values = values.to(example_tensor.device(), example_tensor.dtype()),
    };
  }
};

StandardData generate_data(int n_scenes, int n_samples_per_scene, int n_samples,
                           unsigned base_seed);

struct ImageData {
  StandardData standard;
  at::Tensor image_indexes;

  inline ImageData to(const at::Tensor &example_tensor) const {
    return {
        .standard = standard.to(example_tensor),
        .image_indexes = image_indexes.to(example_tensor.device()),
    };
  }
};

ImageData generate_data_for_image(int n_scenes, int dim, int n_samples,
                                  unsigned base_seed);

// hack to avoid issues with "CUDA free failed: cudaErrorCudartUnloading:
// driver shutting down"
void deinit_renderers();
} // namespace single_triangle
} // namespace generate_data
