#pragma once

#include "render/renderer.h"

#include <torch/extension.h>

#include <tuple>

namespace generate_data {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
gen_data(int n_scenes, int n_samples_per_scene, int n_samples,
         unsigned base_seed);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gen_data_for_image(int n_scenes, int dim, int n_samples, unsigned base_seed);

// needed to avoid issues with "CUDA free failed: cudaErrorCudartUnloading:
// driver shutting down"
void deinit_renderers();
} // namespace generate_data
