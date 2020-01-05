#include "ray/sort_actions.h"
#include "ray/sort_actions_impl.h"
#include "ray/block_data.h"


namespace ray {
namespace detail {
__global__ void sort_actions_impl(Span<const Traversal, false> traversals,
                                  Span<Action> actions) {
  unsigned traversal_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (traversal_idx >= traversals.size()) {
    return;
  }

  sort_traversal_actions(traversals[traversal_idx], actions);
}

void sort_actions(Span<const Traversal, false> traversals,
                  Span<Action> actions) {
  unsigned block_size = 256;
  unsigned blocks = num_blocks(traversals.size(), block_size);

  sort_actions_impl<<<blocks, block_size>>>(traversals, actions);

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}
} // namespace detail
} // namespace ray
