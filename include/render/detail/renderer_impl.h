#pragma once

#include "intersect/accel/accelerator_type.h"
#include "lib/execution_model.h"
#include "lib/execution_model_vector_type.h"
#include "lib/rgba.h"
#include "scene/scene.h"

#include <thrust/device_vector.h>

namespace render {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl {
public:
  void render(RGBA *pixels, const Eigen::Affine3f &film_to_world,
              unsigned x_dim, unsigned y_dim, unsigned samples_per,
              intersect::accel::AcceleratorType mesh_accel_type,
              intersect::accel::AcceleratorType triangle_accel_type,
              bool show_times);

  RendererImpl();

private:
  template <typename T> using ExecVecT = ExecVector<execution_model, T>;
  template <typename T> using SharedVecT = SharedVector<execution_model, T>;

  unsigned x_dim_;
  unsigned y_dim_;
  unsigned samples_per_;

  bool show_times_;

  ExecVecT<Eigen::Vector3f> intensities;
  ExecVecT<RGBA> bgra_;
};
} // namespace detail
} // namespace render
