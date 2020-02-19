#include "render/detail/renderer_impl.h"
#include "render/renderer.h"

namespace render {
using namespace detail;

template <ExecutionModel execution_model>
Renderer<execution_model>::Renderer()
    // needs to not be smart pointer (compiler error otherwise)
    : renderer_impl_(new RendererImpl<execution_model>()) {}

template <ExecutionModel execution_model>
Renderer<execution_model>::~Renderer() {
  delete renderer_impl_;
}

template <ExecutionModel execution_model>
void Renderer<execution_model>::render(
    Span<RGBA> pixels, const scene::Scene &s, unsigned x_dim, unsigned y_dim,
    unsigned samples_per, intersect::accel::AcceleratorType mesh_accel_type,
    intersect::accel::AcceleratorType triangle_accel_type, bool show_times) {
  renderer_impl_->render(pixels, s, x_dim, y_dim, samples_per, mesh_accel_type,
                         triangle_accel_type, show_times);
}

template class Renderer<ExecutionModel::CPU>;
template class Renderer<ExecutionModel::GPU>;
} // namespace render
