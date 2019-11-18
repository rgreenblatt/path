#include "ray/intersect.cuh"

namespace ray {
namespace detail {
__global__ void
solve_intersections(unsigned width, unsigned height, unsigned num_shapes,
                    const scene::ShapeData *shapes,
                    Eigen::Vector4f *world_space_directions,
                    std::optional<BestIntersection> *best_intersections) {
  unsigned x_dim = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y_dim = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned shape_idx = blockIdx.z * blockDim.z + threadIdx.z;

  shapes[shape_idx]
    dkdkf "Ddf


}
} // namespace detail
} // namespace ray
