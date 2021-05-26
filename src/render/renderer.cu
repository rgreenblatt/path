#include "render/detail/renderer_impl.h"
#include "render/renderer.h"

namespace render {
Renderer::Renderer() = default;
Renderer::~Renderer() = default;
Renderer::Renderer(Renderer &&) = default;
Renderer &Renderer::operator=(Renderer &&) = default;

template <typename F>
auto Renderer::visit_renderer(ExecutionModel execution_model, F &&f) {
  switch (execution_model) {
  case ExecutionModel::CPU:
    if (cpu_renderer_impl_ == nullptr) {
      cpu_renderer_impl_ = std::make_unique<Impl<ExecutionModel::CPU>>();
    }

    return f(cpu_renderer_impl_);

  case ExecutionModel::GPU:
#ifdef CPU_ONLY
    std::cerr << "gpu can't be selected for cpu only build" << std::endl;
    abort();
#else
    if (gpu_renderer_impl_ == nullptr) {
      gpu_renderer_impl_ = std::make_unique<Impl<ExecutionModel::GPU>>();
    }

    return f(gpu_renderer_impl_);
#endif

    unreachable();
  };
}

double Renderer::render(ExecutionModel execution_model,
                        const SampleSpec &sample_spec, const Output &output,
                        const scene::Scene &s, unsigned samples_per,
                        const Settings &settings, bool show_progress,
                        bool show_times) {
  return visit_renderer(execution_model, [&](auto &&renderer) {
    return renderer->general_render(sample_spec, output, s, samples_per,
                                    settings, show_progress, show_times);
  });
}
} // namespace render
