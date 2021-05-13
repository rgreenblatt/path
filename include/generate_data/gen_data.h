#pragma once

#include <torch/extension.h>

#include <tuple>

namespace generate_data {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
gen_data(int n_scenes, int n_samples_per_scene, int n_samples);
}

// TODO: rename!
