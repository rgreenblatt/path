#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/enum_accel/enum_accel.h"
#include "lib/bgra.h"
#include "meta/one_per_instance.h"
#include "render/detail/dir_sampler.h"
#include "render/detail/light_sampler.h"
#include "render/detail/term_prob.h"
#include "render/settings.h"
#include "rng/halton.h"
// TODO: generalize...
#include "intersectable_scene/flat_triangle/flat_triangle.h"
#include "rng/sobel.h"
#include "rng/uniform.h"
#include "scene/scene.h"

#include <map>
#include <set>

namespace render {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl {
public:
  void render(Span<BGRA> pixels, const scene::Scene &s, unsigned &samples_per,
              unsigned x_dim, unsigned y_dim, const Settings &settings,
              bool show_progress, bool show_times);

  RendererImpl();

private:
  template <typename T> using ExecVecT = ExecVector<execution_model, T>;
  template <typename T> using SharedVecT = SharedVector<execution_model, T>;

  // TODO: consider eventually freeing...

  using AccelType = intersect::accel::enum_accel::AccelType;

  template <AccelType type>
  using IntersectableSceneGenerator =
      intersectable_scene::flat_triangle::Generator<
          execution_model, intersect::accel::enum_accel::Settings<type>,
          intersect::accel::enum_accel::EnumAccel<type, execution_model>>;

  OnePerInstance<AccelType, IntersectableSceneGenerator>
      stored_scene_generators_;

  template <LightSamplerType type>
  using LightSamplerT = LightSamplerT<type, execution_model>;

  OnePerInstance<LightSamplerType, LightSamplerT> light_samplers_;

  template <DirSamplerType type>
  using DirSamplerT = DirSamplerT<type, execution_model>;

  OnePerInstance<DirSamplerType, DirSamplerT> dir_samplers_;

  template <TermProbType type>
  using TermProbT = TermProbT<type, execution_model>;

  OnePerInstance<TermProbType, TermProbT> term_probs_;

  template <rng::RngType type> using Rng = rng::RngT<type, execution_model>;

  OnePerInstance<rng::RngType, Rng> rngs_;

  ThrustData<execution_model> thrust_data_;

  ExecVecT<Eigen::Array3f> intensities_;
  ExecVecT<scene::TriangleData> triangle_data_;
  ExecVecT<material::Material> materials_;
  ExecVecT<BGRA> bgra_;
};
} // namespace detail
} // namespace render
