#include "lib/unified_memory_vector.h"
#include "ray/intersect.cuh"
#include "ray/render.h"

namespace ray {
unsigned num_blocks(unsigned size, unsigned block_size) {
  return (size + block_size - 1) / block_size;
};

template <typename... T>
void minimize_intersections(unsigned size, T &... values) {
  const unsigned minimize_block_size = 256;
  detail::minimize_intersections<<<num_blocks(size, minimize_block_size),
                                   minimize_block_size>>>(
      size, values.intersections.data()...);
  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template <typename... T>
void minimize_intersections(detail::ByTypeData &first, const T &... rest) {
  minimize_intersections(static_cast<unsigned>(first.intersections.size()),
                         first, rest...);
}

void Renderer::render(const scene::Scene &scene, BGRA *pixels,
                      const scene::Transform &m_film_to_world) {

  for (auto &data : by_type_data_) {
    unsigned num_shapes = scene.get_num_shapes(data.shape_type);

    if (num_shapes == 0) {
      continue;
    }

    const Eigen::Vector3f world_space_eye = m_film_to_world.translation();

    auto shapes = scene.get_shapes(data.shape_type);

#if 1
    // TODO
    /* unsigned width_block_size = 4; */
    /* unsigned height_block_size = 4; */
    /* unsigned shapes_block_size = 64; */

    unsigned width_block_size = 256;
    unsigned height_block_size = 1;

    dim3 grid(num_blocks(width_, width_block_size),
              num_blocks(height_, height_block_size), 1);
    dim3 block(width_block_size, height_block_size, 1);

    detail::solve_intersections<<<grid, block>>>(
        width_, height_, num_shapes, m_film_to_world, world_space_eye, shapes,
        data.intersections.data(), data.shape_type, std::nullopt);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
#else
    detail::solve_intersections_cpu(
        width_, height_, num_shapes, m_film_to_world, world_space_eye, shapes,
        data.intersections.data(), data.shape_type, std::nullopt);
#endif
  }

  minimize_intersections(by_type_data_[0], by_type_data_[1], by_type_data_[2]);

  auto &best_intersections = by_type_data_[0].intersections;

  /* for (unsigned x = 0; x < width_; x++) { */
  /*   for (unsigned y = 0; y < height_; y++) { */
  /*     unsigned index = x + y * width_; */
  /*     if (best_intersections[index].has_value()) { */
  /*       pixels[index] = BGRA(255, 0, 0); */
  /*     } else { */
  /*       pixels[index] = BGRA(0, 0, 0); */
  /*     } */
  /*   } */
  /* } */
}
} // namespace ray
