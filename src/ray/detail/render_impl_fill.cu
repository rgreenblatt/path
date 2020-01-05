#include "ray/detail/impl/fill.h"
#include "ray/detail/render_impl.h"
#include "ray/detail/render_impl_utils.h"

#include <thrust/fill.h>

namespace ray {
namespace detail {

__global__ void initial_world_space_directions_global(
    BlockData block_data, const Eigen::Vector3f world_space_eye,
    const Eigen::Affine3f m_film_to_world,
    Span<Eigen::Vector3f> world_space_directions) {
  initial_world_space_directions_impl(blockIdx.x, threadIdx.x, block_data,
                                      world_space_eye, m_film_to_world,
                                      world_space_directions);
}

// TODO maybe clean this up....
inline void initial_world_space_directions_cpu(
    BlockData block_data, const Eigen::Vector3f &world_space_eye,
    const Eigen::Affine3f &m_film_to_world,
    Span<Eigen::Vector3f> world_space_directions) {
  for (unsigned block_index = 0; block_index < block_data.generalNumBlocks();
       block_index++) {
    for (unsigned thread_index = 0;
         thread_index < block_data.generalBlockSize(); thread_index++) {
      initial_world_space_directions_impl(block_index, thread_index, block_data,
                                          world_space_eye, m_film_to_world,
                                          world_space_directions);
    }
  }
}

template <typename T>
__global__ void fill_data(T *data, unsigned size, T value) {
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] = value;
  }
}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::fill(
    const scene::Color &initial_multiplier, const scene::Color &initial_color,
    const Eigen::Affine3f &m_film_to_world) {
  const unsigned general_num_blocks = block_data_.generalNumBlocks();
  const unsigned general_block_size = block_data_.generalBlockSize();

  const Eigen::Vector3f world_space_eye = m_film_to_world.translation();

  if constexpr (execution_model == ExecutionModel::GPU) {
    const unsigned fill_block_size = 256;
    const unsigned fill_num_blocks =
        num_blocks(block_data_.totalSize(), fill_block_size);

    fill_data<<<fill_num_blocks, fill_block_size>>>(
        to_ptr(world_space_eyes_), block_data_.totalSize(), world_space_eye);
    fill_data<<<fill_num_blocks, fill_block_size>>>(to_ptr(color_multipliers_),
                                                    block_data_.totalSize(),
                                                    initial_multiplier);
    fill_data<<<fill_num_blocks, fill_block_size>>>(
        to_ptr(colors_), block_data_.totalSize(), initial_color);

    initial_world_space_directions_global<<<general_num_blocks,
                                            general_block_size>>>(
        block_data_, world_space_eye, m_film_to_world,
        to_span(world_space_directions_));

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
  } else {
    std::fill(world_space_eyes_.begin(), world_space_eyes_.end(),
              world_space_eye);
    std::fill(color_multipliers_.begin(), color_multipliers_.end(),
              initial_multiplier);
    std::fill(colors_.begin(), colors_.end(), initial_color);

    initial_world_space_directions_cpu(block_data_, world_space_eye,
                                       m_film_to_world,
                                       to_span(world_space_directions_));
  }
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace ray
