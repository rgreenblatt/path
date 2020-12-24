#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "integrate/dir_sampler/enum_dir_sampler/enum_dir_sampler.h"
#include "integrate/light_sampler/enum_light_sampler/enum_light_sampler.h"
#include "integrate/term_prob/enum_term_prob/enum_term_prob.h"
#include "intersect/accel/enum_accel/enum_accel.h"
#include "lib/bgra.h"
#include "meta/one_per_instance.h"
#include "render/settings.h"
#include "rng/enum_rng/enum_rng.h"
// TODO: generalize...
#include "intersectable_scene/flat_triangle/flat_triangle.h"
#include "scene/scene.h"

#include <map>
#include <set>

namespace render {
namespace detail {
using enum_accel::EnumAccel;
using enum_dir_sampler::EnumDirSampler;
using enum_light_sampler::EnumLightSampler;
using enum_term_prob::EnumTermProb;
using rng::enum_rng::EnumRng;

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

  template <AccelType type>
  using IntersectableSceneGenerator =
      intersectable_scene::flat_triangle::Generator<
          execution_model, enum_accel::Settings<type>,
          EnumAccel<type, execution_model>>;

  OnePerInstance<AccelType, IntersectableSceneGenerator>
      stored_scene_generators_;

  template <LightSamplerType type>
  using LightSamplerT = EnumLightSampler<type, execution_model>;

  OnePerInstance<LightSamplerType, LightSamplerT> light_samplers_;

  template <DirSamplerType type> using DirSamplerT = EnumDirSampler<type>;

  OnePerInstance<DirSamplerType, DirSamplerT> dir_samplers_;

  template <TermProbType type> using TermProbT = EnumTermProb<type>;

  OnePerInstance<TermProbType, TermProbT> term_probs_;

  using RngType = rng::enum_rng::RngType;

  template <RngType type>
  using Rng = rng::enum_rng::EnumRng<type, execution_model>;

  OnePerInstance<RngType, Rng> rngs_;

  ThrustData<execution_model> thrust_data_;

  ExecVecT<Eigen::Array3f> intensities_;
  ExecVecT<BGRA> bgra_;
};
} // namespace detail
} // namespace render
