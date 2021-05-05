#pragma once

#include "execution_model/execution_model.h"
#include "lib/bgra_32.h"
#include "lib/float_rgb.h"
#include "lib/span.h"
#include "render/settings.h"
#include "scene/scene.h"

#include <memory>

namespace render {
class Renderer {
public:
  // need to implementated when Impl is defined
  Renderer();
  ~Renderer();
  Renderer(Renderer &&);
  Renderer &operator=(Renderer &&);

  double render(ExecutionModel execution_model, Span<BGRA32> pixels,
                const scene::Scene &s, unsigned samples_per, unsigned x_dim,
                unsigned y_dim, const Settings &settings,
                bool progress_bar = false, bool show_times = false);

  double render_float_rgb(ExecutionModel execution_model,
                          Span<FloatRGB> float_rgb, const scene::Scene &s,
                          unsigned samples_per, unsigned x_dim, unsigned y_dim,
                          const Settings &settings, bool progress_bar = false,
                          bool show_times = false);

private:
  template <typename F>
  double visit_renderer(ExecutionModel execution_model, F &&f);

  template <ExecutionModel execution_model> class Impl;

  std::unique_ptr<Impl<ExecutionModel::CPU>> cpu_renderer_impl_;

  struct EmptyT {};

  // NOTE this makes it invalid to use a renderer in different compilation
  // units with different values of CPU_ONLY
  // The ABI depends on CPU_ONLY
#ifndef CPU_ONLY
  std::unique_ptr<Impl<ExecutionModel::GPU>> gpu_renderer_impl_;
#endif
};
} // namespace render
