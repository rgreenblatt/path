#include "render/detail/renderer_impl.h"
#include "render/renderer.h"

namespace render {
using namespace detail;

Renderer::Renderer() = default;

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer &&) = default;

void Renderer::render(ExecutionModel execution_model, Span<BGRA> pixels,
                      const scene::Scene &s, unsigned samples_per,
                      unsigned x_dim, unsigned y_dim, const Settings &settings,
                      bool show_times) {
  auto render = [&](const auto &renderer) {
    renderer->render(pixels, s, samples_per, x_dim, y_dim, settings,
                     show_times);
  };

  switch (execution_model) {
  case ExecutionModel::CPU:
    if (cpu_renderer_impl_ == nullptr) {
      cpu_renderer_impl_ =
          std::make_unique<RendererImpl<ExecutionModel::CPU>>();
    }

    render(cpu_renderer_impl_);

    return;
  case ExecutionModel::GPU:
#ifdef CPU_ONLY_BUILD
    static_assert(false);
    std::cerr << "gpu can't be selected for cpu only build" << std::endl;
    abort();
#else
    if (gpu_renderer_impl_ == nullptr) {
      gpu_renderer_impl_ =
          std::make_unique<RendererImpl<ExecutionModel::GPU>>();
    }

    render(gpu_renderer_impl_);
#endif

    return;
  };
}
} // namespace render
