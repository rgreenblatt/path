#pragma once

#include "execution_model/execution_model.h"
#include "lib/bgra.h"
#include "lib/span.h"
#include "render/settings.h"
#include "scene/scene.h"

#include <Eigen/Core>

#include <memory>

namespace render {
class Renderer {
public:
  // need to implementated when Impl is defined
  Renderer();
  ~Renderer();
  Renderer(Renderer &&);
  Renderer &operator=(Renderer &&);

  void render(ExecutionModel execution_model, Span<BGRA> pixels,
              const scene::Scene &s, unsigned samples_per, unsigned x_dim,
              unsigned y_dim, const Settings &settings,
              bool progress_bar = false, bool show_times = false);

  void render_intensities(ExecutionModel execution_model,
                          Span<Eigen::Array3f> intensities,
                          const scene::Scene &s, unsigned samples_per,
                          unsigned x_dim, unsigned y_dim,
                          const Settings &settings, bool progress_bar = false,
                          bool show_times = false);

private:
  template <typename F>
  void visit_renderer(ExecutionModel execution_model, F &&f);

  template <ExecutionModel execution_model> class Impl;

  std::unique_ptr<Impl<ExecutionModel::CPU>> cpu_renderer_impl_;
  std::unique_ptr<Impl<ExecutionModel::GPU>> gpu_renderer_impl_;
};
} // namespace render
