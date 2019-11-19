#include "lib/unified_memory_vector.h"
#include "ray/intersect.cuh"
#include "ray/render.h"

namespace ray {
void render(const scene::Scene &scene, BGRA *pixels, unsigned width,
            unsigned height, const scene::Transform &m_film_to_world) {

  // TODO just spheres:::

  scene::Shape shape_type = scene::Shape::Sphere;

  const Eigen::Vector3f camera_space_eye(0, 0, 0);
  const auto world_space_eye = m_film_to_world.linear() * camera_space_eye;

  /* unsigned width_block_size = 4; */
  /* unsigned height_block_size = 4; */
  /* unsigned shapes_block_size = 64; */

  unsigned width_block_size = 16;
  unsigned height_block_size = 16;
  unsigned shapes_block_size = 1; // TODO

  auto num_blocks = [](unsigned size, unsigned block_size) {
    return (size + block_size - 1) / block_size;
  };

  unsigned num_shapes = scene.get_num_shapes(shape_type);
  auto shapes = scene.get_shapes(shape_type);

  dim3 grid(num_blocks(width, width_block_size),
            num_blocks(height, height_block_size),
            num_blocks(num_shapes, shapes_block_size));
  dim3 block(width_block_size, height_block_size, shapes_block_size);

  // TODO on size
  ManangedMemVec<std::optional<detail::BestIntersection>> intersections(width *
                                                                        height);
  ManangedMemVec<Eigen::Vector3f> world_space_direction(width * height);

  detail::solve_intersections<<<grid, block>>>(
      width, height, num_shapes, m_film_to_world, world_space_eye, shapes,
      world_space_direction.data(), intersections.data(), shape_type,
      std::nullopt);

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}
} // namespace ray
