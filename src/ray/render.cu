#include "ray/detail/render_impl.h"
#include "ray/render.h"

namespace ray {
using namespace detail;

template <ExecutionModel execution_model>
Renderer<execution_model>::Renderer(unsigned width, unsigned height,
                                    unsigned super_sampling_rate,
                                    unsigned recursive_iterations,
                                    std::unique_ptr<scene::Scene> &s)
    // needs to not be smart pointer (compiler error otherwise)
    : renderer_impl_(new RendererImpl<execution_model>(
          width, height, super_sampling_rate, recursive_iterations, s)) {}

template <ExecutionModel execution_model>
Renderer<execution_model>::~Renderer() {
  delete renderer_impl_;
}

template <ExecutionModel execution_model>
void Renderer<execution_model>::render(BGRA *pixels,
                                       const Eigen::Affine3f &m_film_to_world,
                                       const Eigen::Projective3f &world_to_film,
                                       bool use_kd_tree, bool use_dir_tree,
                                       bool show_times) {
  renderer_impl_->render(pixels, m_film_to_world, world_to_film, use_kd_tree,
                         use_dir_tree, show_times);
}

template <ExecutionModel execution_model>
scene::Scene &Renderer<execution_model>::get_scene() {
  return renderer_impl_->get_scene();
}

template <ExecutionModel execution_model>
const scene::Scene &Renderer<execution_model>::get_scene() const {
  return renderer_impl_->get_scene();
}

template class Renderer<ExecutionModel::CPU>;
template class Renderer<ExecutionModel::GPU>;
} // namespace ray
