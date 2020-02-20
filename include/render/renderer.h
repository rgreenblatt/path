#pragma once

#include "intersect/accel/accelerator_type.h"
#include "lib/execution_model.h"
#include "lib/rgba.h"
#include "lib/span.h"
#include "render/settings.h"
#include "scene/scene.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl;
}

class Renderer {
public:
  Renderer();

  ~Renderer();

  void render(ExecutionModel execution_model, Span<RGBA> pixels,
              const scene::Scene &s, unsigned samples_per, unsigned x_dim,
              unsigned y_dim, const Settings &settings, bool show_times);

private:
  // needs to not be smart pointer (compiler error otherwise)
  detail::RendererImpl<ExecutionModel::CPU> *cpu_renderer_impl_ = nullptr;
  detail::RendererImpl<ExecutionModel::GPU> *gpu_renderer_impl_ = nullptr;
};
} // namespace render
