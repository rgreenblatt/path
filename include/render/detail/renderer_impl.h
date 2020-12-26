#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "integrate/dir_sampler/enum_dir_sampler/enum_dir_sampler.h"
#include "integrate/light_sampler/enum_light_sampler/enum_light_sampler.h"
#include "integrate/term_prob/enum_term_prob/enum_term_prob.h"
#include "intersect/accel/enum_accel/enum_accel.h"
#include "lib/bgra.h"
#include "meta/one_per_instance.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "rng/enum_rng/enum_rng.h"
// TODO: generalize...
#include "intersectable_scene/flat_triangle/flat_triangle.h"
#include "scene/scene.h"

#include <Eigen/Core>

namespace render {
using enum_accel::EnumAccel;
using enum_dir_sampler::EnumDirSampler;
using enum_light_sampler::EnumLightSampler;
using enum_term_prob::EnumTermProb;
using rng::enum_rng::EnumRng;

template <ExecutionModel exec> class Renderer::Impl {
public:
  Impl();

  void general_render(bool output_as_bgra, Span<BGRA> pixels,
                      Span<Eigen::Array3f> intensities, const scene::Scene &s,
                      unsigned &samples_per, unsigned x_dim, unsigned y_dim,
                      const Settings &settings, bool show_progress,
                      bool show_times);

private:
  template <typename T> using ExecVecT = ExecVector<exec, T>;
  template <typename T> using SharedVecT = SharedVector<exec, T>;

  // TODO: consider eventually freeing...

  template <AccelType type>
  using IntersectableSceneGenerator =
      intersectable_scene::flat_triangle::Generator<
          exec, enum_accel::Settings<type>, EnumAccel<type, exec>>;

  OnePerInstance<AccelType, IntersectableSceneGenerator>
      stored_scene_generators_;

  template <LightSamplerType type>
  using LightSamplerT = EnumLightSampler<type, exec>;

  OnePerInstance<LightSamplerType, LightSamplerT> light_samplers_;

  template <DirSamplerType type> using DirSamplerT = EnumDirSampler<type>;

  OnePerInstance<DirSamplerType, DirSamplerT> dir_samplers_;

  template <TermProbType type> using TermProbT = EnumTermProb<type>;

  OnePerInstance<TermProbType, TermProbT> term_probs_;

  using RngType = rng::enum_rng::RngType;

  template <RngType type> using Rng = rng::enum_rng::EnumRng<type, exec>;

  OnePerInstance<RngType, Rng> rngs_;

  ThrustData<exec> thrust_data_;

  ExecVecT<Eigen::Array3f> intensities_;
  ExecVecT<Eigen::Array3f> reduced_intensities_;
  ExecVecT<BGRA> bgra_;
};
} // namespace render
