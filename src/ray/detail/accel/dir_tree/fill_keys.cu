#include "lib/async_for.h"
#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "ray/detail/accel/dir_tree/group.h"
#include "ray/detail/block_data.h"

#include <cmath>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
__global__ void fill_keys_global(SpanSized<unsigned> keys,
                                 unsigned num_elements_per_thread,
                                 SpanSized<unsigned> groups, uint8_t axis) {
  // blockDim is divisible by warpSize
  unsigned element_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_idx = element_idx / warpSize;
  unsigned lane = threadIdx.x % warpSize;
  unsigned group_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (group_idx >= groups.size()) {
    return;
  }
  auto [start, end] = group_start_end(group_idx, groups);
  unsigned element_start = lane + warp_idx * num_elements_per_thread + start;
  unsigned element_end =
      std::min(end, element_start + warpSize * num_elements_per_thread);

  for (unsigned i = element_start; i < element_end; i += warpSize) {
    keys[i] = group_idx;
  }
}

template <> void DirTreeGenerator<ExecutionModel::GPU>::fill_keys() {
  // perhaps check elements per thread is bigger than size and
  // maybe have divisions per thread...
  const unsigned num_elements_per_thread = 4;
  const unsigned block_total_size = 512;

  std::array<SpanSized<unsigned>, 3> keys = {x_edges_keys_, y_edges_keys_,
                                             z_keys_};

  std::array<unsigned, 3> max_sizes;
  async_for<true>(0, 3, [&](unsigned axis) {
    auto group = groups_[axis];

    max_sizes[axis] = thrust::transform_reduce(
        thrust_data_[0].execution_policy(), thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(unsigned(group.size())),
        [group] __host__ __device__(const unsigned j) {
          return group_size(j, group);
        },
        0u,
        [] __host__ __device__(const unsigned first, const unsigned second) {
          return std::max(first, second);
        });
  });

#if 0
  async_for<true>(0, 3, [&](unsigned axis) {
    unsigned block_size_elements =
        (std::min(block_total_size, max_sizes[axis] / num_elements_per_thread) /
         32) *
        32;
    unsigned block_size_divisions = block_total_size / block_size_elements;

    dim3 grid(num_blocks(max_sizes[axis], block_size_elements),
              num_blocks(divisions.size(), block_size_divisions));
    dim3 block(block_size_elements, block_size_divisions);

    fill_keys_global<<<grid, block>>>(keys[axis], num_elements_per_thread,
                                      divisions, axis);
  });
#endif
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
