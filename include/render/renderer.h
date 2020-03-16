#pragma once

#include "execution_model/execution_model.h"
#include "intersect/accel/accel.h"
#include "lib/bgra.h"
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

  Renderer(const Renderer &) = delete;

  Renderer(Renderer &&);

  ~Renderer();

  void render(ExecutionModel execution_model, Span<BGRA> pixels,
              thrust::optional<Span<Eigen::Array3f>> intensities,
              thrust::optional<Span<Eigen::Array3f>> variances,
              const scene::Scene &s, unsigned samples_per, unsigned x_dim,
              unsigned y_dim, const Settings &settings,
              bool show_times = false);

private:
  std::unique_ptr<detail::RendererImpl<ExecutionModel::CPU>> cpu_renderer_impl_;
  std::unique_ptr<detail::RendererImpl<ExecutionModel::GPU>> gpu_renderer_impl_;
};
} // namespace render
