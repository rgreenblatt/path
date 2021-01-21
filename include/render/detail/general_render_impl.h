#pragma once

#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/triangle_impl.h"
#include "intersectable_scene/to_bulk_impl.h"
#include "kernel/work_division.h"
#include "meta/all_values/dispatch.h"
#include "render/detail/integrate_image.h"
#include "render/detail/reduce_float_rgb.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/settings_compile_time_impl.h"

namespace render {
using namespace detail;

template <ExecutionModel exec>
void Renderer::Impl<exec>::general_render(
    bool output_as_bgra_32, Span<BGRA32> bgra_32_output,
    Span<FloatRGB> float_rgb_output, const scene::Scene &s,
    unsigned samples_per, unsigned x_dim, unsigned y_dim,
    const Settings &settings, bool show_progress, bool) {
  WorkDivision division = WorkDivision(
      settings.general_settings.computation_settings.render_work_division,
      samples_per, x_dim, y_dim);

  // We could save a copy in the cpu case by directly outputing to the input
  // bgra_32/float_rgb when possible.
  // However, the perf gains are small and we don't really
  // case about the performance of the cpu use case.

  if (division.num_sample_blocks() != 1 || !output_as_bgra_32) {
    float_rgb_.resize(division.num_sample_blocks() * x_dim * y_dim);
  }

  if (output_as_bgra_32) {
    bgra_32_.resize(x_dim * y_dim);
  }

  dispatch(settings.compile_time(), [&](auto tag) {
    constexpr auto compile_time = decltype(tag)::value;

    constexpr auto intersection_type = compile_time.intersection_type;
    constexpr auto light_sampler_type = compile_time.light_sampler_type;
    constexpr auto dir_sampler_type = compile_time.dir_sampler_type;
    constexpr auto term_prob_type = compile_time.term_prob_type;
    constexpr auto rng_type = compile_time.rng_type;

    // this will need to change somewhat... when another value is added...
    static_assert(AllValues<IntersectionApproach>.size() == 2);
    constexpr auto intersection_approach = intersection_type.type();
    constexpr auto accel_type =
        intersection_type.get(TagV<intersection_approach>);

    const auto &all_intersection_settings =
        settings.intersection.get(TagV<intersection_approach>);
    const auto &accel_settings = [&]() -> const auto & {
      if constexpr (intersection_approach == IntersectionApproach::MegaKernel) {
        return all_intersection_settings;
      } else if constexpr (intersection_approach ==
                           IntersectionApproach::StreamingFromGeneral) {
        static_assert(intersection_approach ==
                      IntersectionApproach::StreamingFromGeneral);
        return all_intersection_settings.accel;
      }
    }
    ().get(TagV<accel_type>);

    auto intersectable_scene =
        stored_scene_generators_.get(TagV<accel_type>).gen({accel_settings}, s);

    decltype(auto) intersector = [&]() -> decltype(auto) {
      if constexpr (intersection_approach == IntersectionApproach::MegaKernel) {
        return intersectable_scene.intersector;
      } else if constexpr (intersection_approach ==
                           IntersectionApproach::StreamingFromGeneral) {
        auto &out = to_bulk_.get(TagV<accel_type>);
        out.set_settings_intersectable(
            all_intersection_settings.to_bulk_settings,
            intersectable_scene.intersector);

        return out;
      }
    }();

    auto light_sampler =
        light_samplers_.get(TagV<light_sampler_type>)
            .gen(settings.light_sampler.get(TagV<light_sampler_type>),
                 s.emissive_clusters(), s.emissive_cluster_ends_per_mesh(),
                 s.materials().as_unsized(), s.transformed_mesh_objects(),
                 s.transformed_mesh_idxs(), s.triangles().as_unsized());

    auto dir_sampler =
        dir_samplers_.get(TagV<dir_sampler_type>)
            .gen(settings.dir_sampler.get(TagV<dir_sampler_type>));

    auto term_prob = term_probs_.get(TagV<term_prob_type>)
                         .gen(settings.term_prob.get(TagV<term_prob_type>));

    unsigned n_locations = x_dim * y_dim;

    auto rng =
        rngs_.get(TagV<rng_type>)
            .gen(settings.rng.get(TagV<rng_type>), samples_per, n_locations);

    decltype(auto) state = [&]() -> decltype(auto) {
      if constexpr (intersection_approach == IntersectionApproach::MegaKernel) {
        return IntegrateImageEmptyState{};
      } else {
        unreachable();
        static_assert(intersection_approach ==
                      IntersectionApproach::StreamingFromGeneral);
        return bulk_state_.get(
            TagV<make_meta_tuple(light_sampler_type, rng_type)>);
      }
    }();

    // not really any good way to infer these arguments...
    using Components = integrate::RenderingEquationComponents<
        decltype(intersectable_scene.scene), decltype(light_sampler),
        decltype(dir_sampler), decltype(term_prob)>;
    using Items = IntegrateImageItems<Components, decltype(rng)>;

    IntegrateImage<exec>::template run<Items, decltype(intersector)>({{
        .items =
            {
                .base =
                    {
                        .output_as_bgra_32 = output_as_bgra_32,
                        .samples_per = samples_per,
                        .bgra_32 = bgra_32_,
                        .float_rgb = float_rgb_,
                    },
                .components =
                    {
                        .scene = intersectable_scene.scene,
                        .light_sampler = light_sampler,
                        .dir_sampler = dir_sampler,
                        .term_prob = term_prob,
                    },
                .rng = rng,
                .film_to_world = s.film_to_world(),
            },
        .intersector = intersector,
        .division = division,
        .settings = settings.general_settings,
        .show_progress = show_progress,
        .state = state,
    }});
  });

#if 1
  auto float_rgb_reduce_out = ReduceFloatRGB<exec>::run(
      output_as_bgra_32, division.num_sample_blocks(), samples_per, &float_rgb_,
      &reduced_float_rgb_, bgra_32_);
  always_assert(float_rgb_reduce_out != nullptr);
  always_assert(float_rgb_reduce_out == &float_rgb_ ||
                float_rgb_reduce_out == &reduced_float_rgb_);

  if (output_as_bgra_32) {
    thrust::copy(bgra_32_.begin(), bgra_32_.end(), bgra_32_output.begin());
  } else {
    thrust::copy(float_rgb_reduce_out->begin(), float_rgb_reduce_out->end(),
                 float_rgb_output.begin());
  }
#endif
}
} // namespace render
