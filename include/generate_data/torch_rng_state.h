#include "rng/rng.h"

#include <torch/extension.h>

namespace generate_data {
struct TorchRng {
  // TODO: is this efficient???
  float next() { return torch::rand(1).template item<float>(); }
};

static_assert(rng::RngState<TorchRng>);
} // namespace generate_data
