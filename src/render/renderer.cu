#include "render/detail/renderer_impl.h"
#include "render/renderer.h"

namespace render {
Renderer::Renderer() = default;
Renderer::~Renderer() = default;
Renderer::Renderer(Renderer &&) = default;
Renderer &Renderer::operator=(Renderer &&) = default;

template <typename F>
void Renderer::visit_renderer(ExecutionModel execution_model, F &&f) {
  switch (execution_model) {
  case ExecutionModel::CPU:
    if (cpu_renderer_impl_ == nullptr) {
      cpu_renderer_impl_ = std::make_unique<Impl<ExecutionModel::CPU>>();
    }

    f(cpu_renderer_impl_);

    return;
  case ExecutionModel::GPU:
#ifdef CPU_ONLY
    std::cerr << "gpu can't be selected for cpu only build" << std::endl;
    abort();
#else
    if (gpu_renderer_impl_ == nullptr) {
      gpu_renderer_impl_ = std::make_unique<Impl<ExecutionModel::GPU>>();
    }

    f(gpu_renderer_impl_);
#endif

    return;
  };
}

void Renderer::render(ExecutionModel execution_model, Span<BGRA> pixels,
                      const scene::Scene &s, unsigned samples_per,
                      unsigned x_dim, unsigned y_dim, const Settings &settings,
                      bool show_progress, bool show_times) {
  visit_renderer(execution_model, [&](auto &&renderer) {
    renderer->general_render(true, pixels, {}, s, samples_per, x_dim, y_dim,
                             settings, show_progress, show_times);
  });
}

void Renderer::render_intensities(ExecutionModel execution_model,
                                  Span<Eigen::Array3f> intensities,
                                  const scene::Scene &s, unsigned samples_per,
                                  unsigned x_dim, unsigned y_dim,
                                  const Settings &settings, bool show_progress,
                                  bool show_times) {
  visit_renderer(execution_model, [&](auto &&renderer) {
    renderer->general_render(false, {}, intensities, s, samples_per, x_dim,
                             y_dim, settings, show_progress, show_times);
  });
}
} // namespace render
