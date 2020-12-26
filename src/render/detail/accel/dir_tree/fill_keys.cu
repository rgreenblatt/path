#include "lib/async_for.h"
#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/timer.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/group.h"
#include "ray/detail/block_data.h"

#include <cmath>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
__global__ void fill_keys_global(SpanSized<unsigned> keys,
                                 unsigned num_elements_per_thread,
                                 SpanSized<unsigned> groups) {
  // blockDim is divisible by warp_size
  unsigned element_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_idx = element_idx / warp_size;
  unsigned lane = threadIdx.x % warp_size;
  unsigned group_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (group_idx >= groups.size()) {
    return;
  }
  auto [start, end] = group_start_end(group_idx, groups);
  unsigned element_start = lane + warp_idx * num_elements_per_thread + start;
  unsigned element_end =
      std::min(end, element_start + warp_size * num_elements_per_thread);

  for (unsigned i = element_start; i < element_end; i += warp_size) {
    keys[i] = group_idx;
  }
}

template <> void DirTreeGeneratorImpl<ExecutionModel::GPU>::fill_keys() {
  Timer max_size_timer;

  std::array<unsigned, 3> max_sizes;
  async_for(use_async_, 0, 3, [&](unsigned axis) {
    auto groups = axis_groups_.first.get()[axis];

    max_sizes[axis] = thrust::transform_reduce(
        thrust_data_[0].execution_policy(), thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(unsigned(axis_groups_.first->size())),
        [groups] __host__ __device__(const unsigned j) {
          return group_size(j, groups);
        },
        0u,
        [] __host__ __device__(const unsigned first, const unsigned second) {
          return std::max(first, second);
        });
  });

  max_size_timer.report("max size for fill keys");

  // perhaps check elements per thread is bigger than size and
  // maybe have divisions per thread...
  const unsigned num_elements_per_thread = 4;
  const unsigned block_total_size = 512;

  std::array<SpanSized<unsigned>, 3> keys = {x_edges_keys_, y_edges_keys_,
                                             z_keys_};

  Timer fill_keys_timer;

  async_for(use_async_, 0, 3, [&](unsigned axis) {
    unsigned block_size_elements =
        (std::min(block_total_size, max_sizes[axis] / num_elements_per_thread) /
         32) *
        32;
    unsigned block_size_groups = block_total_size / block_size_elements;

    dim3 grid(num_blocks(max_sizes[axis], block_size_elements),
              num_blocks(num_groups(), block_size_groups));
    dim3 block(block_size_elements, block_size_groups);

    fill_keys_global<<<grid, block>>>(keys[axis], num_elements_per_thread,
                                      axis_groups_.first.get()[axis]);
  });

  fill_keys_timer.report("fill keys");
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
