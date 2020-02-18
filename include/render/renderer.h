#pragma once

#include "lib/rgba.h"
#include "lib/execution_model.h"
#include "scene/scene.h"
#include "intersect/accel/accelerator_type.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl;
}

template <ExecutionModel execution_model> class Renderer {
public:
  Renderer();

  ~Renderer();

  void render(RGBA *pixels, const Eigen::Affine3f &film_to_world,
              unsigned x_dim, unsigned y_dim, unsigned samples_per,
              intersect::accel::AcceleratorType mesh_accel_type,
              intersect::accel::AcceleratorType triangle_accel_type,
              bool show_times);

private:
  // needs to not be smart pointer (compiler error otherwise)
  detail::RendererImpl<execution_model> *renderer_impl_;
};
} // namespace render
