#pragma once

#include "execution_model/execution_model.h"
#include "lib/bgra_32.h"
#include "lib/float_rgb.h"
#include "lib/span.h"
#include "render/settings.h"
#include "scene/scene.h"

#include <memory>

namespace render {
enum class OutputType {
  BGRA,
  FloatRGB,
  OutputPerStep,
};

enum class SampleSpecType {
  SquareImage,
  InitialRays,
};

struct SquareImageSpec {
  unsigned x_dim;
  unsigned y_dim;
  Eigen::Affine3f film_to_world;
};

using Output = TaggedUnion<OutputType, Span<BGRA32>, Span<FloatRGB>,
                           SpanSized<const Span<FloatRGB>>>;
using SampleSpec = TaggedUnion<SampleSpecType, SquareImageSpec,
                               SpanSized<const intersect::Ray>>;

class Renderer {
public:
  // need to implementated when Impl is defined
  Renderer();
  ~Renderer();
  Renderer(Renderer &&);
  Renderer &operator=(Renderer &&);

  double render(ExecutionModel execution_model, const SampleSpec &sample_spec,
                const Output &output, const scene::Scene &s,
                unsigned samples_per, const Settings &settings,
                bool show_progress, bool show_times = false);

private:
  template <typename F>
  auto visit_renderer(ExecutionModel execution_model, F &&f);

  template <ExecutionModel execution_model> class Impl;

  std::unique_ptr<Impl<ExecutionModel::CPU>> cpu_renderer_impl_;

  // NOTE this makes it invalid to use a renderer in different compilation
  // units with different values of CPU_ONLY
  // The ABI depends on CPU_ONLY
#ifndef CPU_ONLY
  std::unique_ptr<Impl<ExecutionModel::GPU>> gpu_renderer_impl_;
#endif
};
} // namespace render
