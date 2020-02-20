#include "render/detail/renderer_impl.h"
#include "render/renderer.h"

namespace render {
using namespace detail;

Renderer::Renderer() {}

Renderer::~Renderer() {
  if (cpu_renderer_impl_ != nullptr) {
    delete cpu_renderer_impl_;
  }
  if (gpu_renderer_impl_ != nullptr) {
    delete gpu_renderer_impl_;
  }
}

void Renderer::render(ExecutionModel execution_model, Span<RGBA> pixels,
                      const scene::Scene &s, unsigned samples_per,
                      unsigned x_dim, unsigned y_dim, PerfSettings settings,
                      bool show_times) {
  auto render = [&](auto renderer) {
    renderer->render(pixels, s, samples_per, x_dim, y_dim, settings,
                     show_times);
  };

  switch (execution_model) {
  case ExecutionModel::CPU:
    if (cpu_renderer_impl_ == nullptr) {
      cpu_renderer_impl_ = new RendererImpl<ExecutionModel::CPU>();
    }

    render(cpu_renderer_impl_);
  case ExecutionModel::GPU:
    if (gpu_renderer_impl_ == nullptr) {
      gpu_renderer_impl_ = new RendererImpl<ExecutionModel::GPU>();
    }

    render(gpu_renderer_impl_);
  };
}
} // namespace render
