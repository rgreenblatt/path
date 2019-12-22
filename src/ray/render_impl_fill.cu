#include "ray/render_impl.h"
#include "ray/render_impl_utils.h"

namespace ray {
using namespace detail;
inline HOST_DEVICE Eigen::Vector3f
initial_world_space_direction(unsigned x, unsigned y, unsigned x_dim,
                              unsigned y_dim,
                              const Eigen::Vector3f &world_space_eye,
                              const Eigen::Affine3f &m_film_to_world) {
  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * static_cast<float>(x)) / static_cast<float>(x_dim) - 1.0f,
      (-2.0f * static_cast<float>(y)) / static_cast<float>(y_dim) + 1.0f,
      -1.0f);
  const auto world_space_film_plane = m_film_to_world * camera_space_film_plane;

  return (world_space_film_plane - world_space_eye).normalized();
}

__inline__ __host__ __device__ void initial_world_space_directions_impl(
    unsigned block_index, unsigned thread_index, const BlockData &block_data,
    const Eigen::Vector3f &world_space_eye,
    const scene::Transform &m_film_to_world,
    Span<Eigen::Vector3f> world_space_directions) {
  auto [x, y, index, outside_bounds] =
      block_data.getIndexes(block_index, thread_index);

  if (outside_bounds) {
    return;
  }

  world_space_directions[index] =
      initial_world_space_direction(x, y, block_data.x_dim, block_data.y_dim,
                                    world_space_eye, m_film_to_world);
}

__global__ void
initial_world_space_directions(BlockData block_data,
                               const Eigen::Vector3f world_space_eye,
                               const scene::Transform m_film_to_world,
                               Span<Eigen::Vector3f> world_space_directions) {
  initial_world_space_directions_impl(blockIdx.x, threadIdx.x, block_data,
                                      world_space_eye, m_film_to_world,
                                      world_space_directions);
}

inline void initial_world_space_directions_cpu(
    BlockData block_data, const Eigen::Vector3f &world_space_eye,
    const scene::Transform &m_film_to_world,
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
    const Eigen::Affine3f &m_film_to_world) {
  const scene::Color initial_multiplier = scene::Color::Ones();
  const scene::Color initial_color = scene::Color::Zero();
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

    initial_world_space_directions<<<general_num_blocks, general_block_size>>>(
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

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace ray
