#include "ray/intersect.cuh"
#include "ray/render.h"

namespace ray {
void render(const scene::Scene &scene, BGRA *pixels) {
  dim3 grid(1, 1, 1);
  dim3 block(1, 1, 1);
  detail::solve_intersections<<<grid, block>>>(100, 100, scene.num_cubes(), scene.cubes(), nullptr,
                      nullptr);
}
} // namespace ray
