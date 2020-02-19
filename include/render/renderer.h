#pragma once

#include "intersect/accel/accelerator_type.h"
#include "lib/execution_model.h"
#include "render/settings.h"
#include "lib/rgba.h"
#include "lib/span.h"
#include "scene/scene.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl;
}

template <ExecutionModel execution_model> class Renderer {
public:
  Renderer();

  ~Renderer();

  void render(Span<RGBA> pixels, const scene::Scene &s, unsigned x_dim,
              unsigned y_dim, unsigned samples_per, PerfSettings settings,
              bool show_times);

private:
  // needs to not be smart pointer (compiler error otherwise)
  detail::RendererImpl<execution_model> *renderer_impl_;
};
} // namespace render
