#include "ray/intersect.cuh"
#include "ray/render.h"

namespace ray {
void render(const scene::Scene &scene, BGRA *pixels) {
  solve_intersections(100, 100, scene.num_cubes(), scene.cubes(), nullptr,
                      nullptr);
}
} // namespace ray
