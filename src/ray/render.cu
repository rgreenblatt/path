#include "ray/render.h"
#include "ray/render_impl.h"

namespace ray {
template <ExecutionModel execution_model>
Renderer<execution_model>::Renderer(unsigned width, unsigned height,
                                    unsigned super_sampling_rate,
                                    unsigned recursive_iterations)
    : renderer_impl_(new RendererImpl<execution_model>(
          width, height, super_sampling_rate, recursive_iterations,
          // TODO*****
          Eigen::Vector3f(), Eigen::Vector3f())) {}

template <ExecutionModel execution_model>
Renderer<execution_model>::~Renderer() {
  delete renderer_impl_;
}

template <ExecutionModel execution_model>
void Renderer<execution_model>::render(const scene::Scene &scene, BGRA *pixels,
                                       const scene::Transform &m_film_to_world,
                                       const Eigen::Projective3f &world_to_film,
                                       bool use_kd_tree, bool show_times) {
  renderer_impl_->render(scene, pixels, m_film_to_world, world_to_film,
                         use_kd_tree, show_times);
}

template class Renderer<ExecutionModel::CPU>;
template class Renderer<ExecutionModel::GPU>;
} // namespace ray
