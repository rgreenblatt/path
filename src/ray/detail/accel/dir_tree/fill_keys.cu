#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "ray/detail/block_data.h"
#include <cmath>

#include <future>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
__global__ void fill_keys_global(SpanSized<unsigned> keys,
                                 unsigned num_elements_per_thread,
                                 SpanSized<const WorkingDivision> divisions,
                                 uint8_t axis) {
  // blockDim is divisible by warpSize
  unsigned element_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_idx = element_idx / warpSize;
  unsigned lane = threadIdx.x % warpSize;
  unsigned division_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (division_idx >= divisions.size()) {
    return;
  }
  const auto &division = divisions[division_idx];
  unsigned element_start =
      lane + warp_idx * num_elements_per_thread + division.starts[axis];
  unsigned element_end = std::min(
      division.ends[axis], element_start + warpSize * num_elements_per_thread);

  for (unsigned i = element_start; i < element_end; i += warpSize) {
    keys[i] = division_idx;
  }
}

template <> void DirTreeGenerator<ExecutionModel::GPU>::fill_keys() {
  // perhaps check elements per thread is bigger than size and
  // maybe have divisions per thread...
  const unsigned num_elements_per_thread = 4;
  const unsigned block_total_size = 512;

  SpanSized<const WorkingDivision> divisions(divisions_);

  std::array<SpanSized<unsigned>, 3> keys = {x_edges_keys_, y_edges_keys_,
                                             z_keys_};

  std::array<unsigned, 3> max_sizes = thrust::transform_reduce(
      thrust_data_[0].execution_policy(), divisions_.begin(), divisions_.end(),
      [] __host__ __device__(const WorkingDivision &div) {
        return div.sizes();
      },
      std::array<unsigned, 3>{0, 0, 0},
      [] __host__ __device__(const std::array<unsigned, 3> &first,
                             const std::array<unsigned, 3> &second) {
        return std::array<unsigned, 3>{std::min(first[0], second[0]),
                                       std::min(first[1], second[1]),
                                       std::min(first[2], second[2])};
      });

  for (uint8_t axis = 0; axis < 3; axis++) {
    std::async(std::launch::async, [&] {
      unsigned block_size_elements =
          (std::min(block_total_size,
                    max_sizes[axis] / num_elements_per_thread) /
           32) *
          32;
      unsigned block_size_divisions = block_total_size / block_size_elements;

      dim3 grid(num_blocks(max_sizes[axis], block_size_elements),
                num_blocks(divisions.size(), block_size_divisions));
      dim3 block(block_size_elements, block_size_divisions);

      fill_keys_global<<<grid, block>>>(keys[axis], num_elements_per_thread,
                                        divisions, axis);
    });
  }
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
