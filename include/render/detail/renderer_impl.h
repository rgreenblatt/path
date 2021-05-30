#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "integrate/dir_sampler/enum_dir_sampler/enum_dir_sampler.h"
#include "integrate/light_sampler/enum_light_sampler/enum_light_sampler.h"
#include "integrate/term_prob/enum_term_prob/enum_term_prob.h"
#include "intersect/accel/enum_accel/enum_accel.h"
#include "intersectable_scene/flat_triangle/flat_triangle.h"
#include "intersectable_scene/to_bulk.h"
#include "lib/tagged_tuple.h"
#include "meta/all_values/impl/enum.h"
#include "render/detail/integrate_image/streaming/state.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "rng/enum_rng/enum_rng.h"
#include "scene/scene.h"

namespace render {
using enum_accel::EnumAccel;
using enum_dir_sampler::EnumDirSampler;
using enum_light_sampler::EnumLightSampler;
using enum_term_prob::EnumTermProb;
using rng::enum_rng::EnumRng;

template <ExecutionModel exec> class Renderer::Impl {
public:
  Impl();

  // returns execution time (not including build time)
  double general_render(const SampleSpec &sample_spec, const Output &output,
                        const scene::Scene &s, unsigned samples_per,
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

  TaggedTuplePerInstance<AccelType, IntersectableSceneGenerator>
      stored_scene_generators_;

  template <LightSamplerType type>
  using LightSamplerT = EnumLightSampler<type, exec>;

  TaggedTuplePerInstance<LightSamplerType, LightSamplerT> light_samplers_;

  template <DirSamplerType type> using DirSamplerT = EnumDirSampler<type>;

  TaggedTuplePerInstance<DirSamplerType, DirSamplerT> dir_samplers_;

  template <TermProbType type> using TermProbT = EnumTermProb<type>;

  TaggedTuplePerInstance<TermProbType, TermProbT> term_probs_;

  using RngType = rng::enum_rng::RngType;

  template <RngType type> using Rng = rng::enum_rng::EnumRng<type, exec>;

  TaggedTuplePerInstance<RngType, Rng> rngs_;

  template <AccelType type>
  using BulkIntersectableSceneGenerator = intersectable_scene::ToBulkGen<
      exec, typename IntersectableSceneGenerator<type>::Intersector>;

  TaggedTuplePerInstance<AccelType, BulkIntersectableSceneGenerator> to_bulk_;

  using StreamingStateType = MetaTuple<LightSamplerType, RngType>;

  template <StreamingStateType type>
  using IntegrateImageStreamingState =
      detail::integrate_image::streaming::State<
          exec, LightSamplerT<meta_tuple_at<0>(type)>::Ref::max_num_samples,
          typename Rng<meta_tuple_at<1>(type)>::Ref>;

  TaggedTuplePerInstance<StreamingStateType, IntegrateImageStreamingState>
      streaming_state_;

  ExecVecT<BGRA32> bgra_32_;
  std::array<ExecVecT<FloatRGB>, 2> float_rgb_;
  std::array<HostVector<ExecVecT<FloatRGB>>, 2> output_per_step_rgb_;
  std::array<HostVector<Span<FloatRGB>>, 2> output_per_step_rgb_spans_;
  std::array<ExecVecT<Span<FloatRGB>>, 2> output_per_step_rgb_spans_device_;
  HostVector<Span<FloatRGB>> output_per_step_rgb_spans_out_;

  ExecVecT<intersect::Ray> sample_rays_;
  ExecVecT<InitialIdxAndDirSpec> sample_idxs_and_dir_;

  ThrustData<exec> thrust_data_;
};
} // namespace render
