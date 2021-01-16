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
#include "render/detail/integrate_image_bulk_state.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "rng/enum_rng/enum_rng.h"
#include "scene/scene.h"
#include "meta/all_values_enum.h"

namespace render {
using enum_accel::EnumAccel;
using enum_dir_sampler::EnumDirSampler;
using enum_light_sampler::EnumLightSampler;
using enum_term_prob::EnumTermProb;
using rng::enum_rng::EnumRng;

template <ExecutionModel exec> class Renderer::Impl {
public:
  Impl();

  void general_render(bool output_as_bgra_32, Span<BGRA32> bgra_32_output,
                      Span<FloatRGB> float_rgb_output, const scene::Scene &s,
                      unsigned samples_per, unsigned x_dim, unsigned y_dim,
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

  using BulkStateType = MetaTuple<LightSamplerType, RngType>;

  template <BulkStateType type>
  using IntegrateImageBulkState = detail::IntegrateImageBulkState<
      exec, LightSamplerT<meta_tuple_at<0>(type)>::Ref::max_num_samples,
      typename Rng<meta_tuple_at<1>(type)>::Ref>;

  TaggedTuplePerInstance<BulkStateType, IntegrateImageBulkState> bulk_state_;

  ThrustData<exec> thrust_data_;

  ExecVecT<FloatRGB> float_rgb_;
  ExecVecT<FloatRGB> reduced_float_rgb_;
  ExecVecT<BGRA32> bgra_32_;
};
} // namespace render
