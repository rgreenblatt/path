#pragma once

#include "generate_data/generate_data.h"

#include <ATen/Tensor.h>

#include <optional>
#include <tuple>
#include <vector>

namespace generate_data {
namespace full_scene {
struct NetworkInputs {
  at::Tensor triangle_features;
  at::Tensor mask;
  at::Tensor bsdf_features;
  at::Tensor emissive_values;
  at::Tensor baryocentric_coords;
  at::Tensor triangle_idxs_for_coords;
  unsigned total_tri_count;
  std::vector<unsigned> n_samples_per;

  inline NetworkInputs to(const at::Tensor &example_tensor) const {
    return {
        .triangle_features = triangle_features.to(example_tensor.device(),
                                                  example_tensor.dtype()),
        .mask = mask.to(example_tensor.device()),
        .bsdf_features =
            bsdf_features.to(example_tensor.device(), example_tensor.dtype()),
        .emissive_values =
            emissive_values.to(example_tensor.device(), example_tensor.dtype()),
        .baryocentric_coords = baryocentric_coords.to(example_tensor.device(),
                                                      example_tensor.dtype()),
        .triangle_idxs_for_coords =
            triangle_idxs_for_coords.to(example_tensor.device()),
        .total_tri_count = total_tri_count,
        .n_samples_per = n_samples_per,
    };
  }
};

StandardData<NetworkInputs> generate_data(int max_tris,
                                          std::optional<int> forced_n_scenes,
                                          int n_samples_per_tri, int n_samples,
                                          int n_steps, unsigned base_seed);

ImageData<NetworkInputs>
generate_data_for_image(int max_tris, std::optional<int> forced_n_scenes,
                        int dim, int n_samples, int n_steps,
                        unsigned base_seed);

// hack to avoid issues with "CUDA free failed: cudaErrorCudartUnloading:
// driver shutting down"
void deinit_renderers();
} // namespace full_scene
} // namespace generate_data
