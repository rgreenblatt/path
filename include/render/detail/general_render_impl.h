#pragma once

#include "integrate/rendering_equation_components.h"
#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/triangle_impl.h"
#include "intersectable_scene/to_bulk_impl.h"
#include "kernel/work_division.h"
#include "meta/all_values/dispatch.h"
#include "render/detail//integrate_image/mega_kernel/run.h"
#include "render/detail//integrate_image/streaming/run.h"
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
  if (output_as_bgra_32) {
    bgra_32_.resize(x_dim * y_dim);
  }

  // We return float_rgb_out because the mega kernel reduce can pick
  // which value to avoid a copy. (might be premature optimization...)
  auto float_rgb_out = dispatch(settings.compile_time(), [&](auto tag) {
    constexpr auto compile_time = tag();

    constexpr auto kernel_approach_type = compile_time.kernel_approach_type;
    constexpr auto light_sampler_type = compile_time.light_sampler_type;
    constexpr auto dir_sampler_type = compile_time.dir_sampler_type;
    constexpr auto term_prob_type = compile_time.term_prob_type;
    constexpr auto rng_type = compile_time.rng_type;

    constexpr auto kernel_approach = kernel_approach_type.type();

    auto as_intersectable_scene =
        [&](auto accel_type, IndividuallyIntersectableSettings settings) {
          auto accel_settings = settings.accel.get(accel_type);

          return stored_scene_generators_.get(accel_type)
              .gen({accel_settings}, s);
        };

    constexpr auto kernel_approach_type_value =
        kernel_approach_type.get(tag_v<kernel_approach>);
    const auto &kernel_approach_settings =
        settings.kernel_approach.get(tag_v<kernel_approach>);
    auto intersectable_items = [&]() {
      if constexpr (kernel_approach == KernelApproach::MegaKernel) {
        constexpr auto accel_type = kernel_approach_type_value;
        auto intersectable_scene = as_intersectable_scene(
            tag_v<accel_type>,
            kernel_approach_settings.individually_intersectable_settings);
        return std::tuple{intersectable_scene.scene,
                          intersectable_scene.intersector};
      } else {
        static_assert(kernel_approach == KernelApproach::Streaming);
        auto &accel = kernel_approach_settings.accel;
        return accel.visit_tagged([&](auto tag, const auto &value) {
          // If more cases are added, this will need to be an set of if else
          // statements.
          static_assert(
              tag ==
              StreamingSettings::BulkIntersectionApproaches::IndividualToBulk);

          constexpr auto accel_type = kernel_approach_type_value.get(tag);
          auto intersectable_scene = as_intersectable_scene(
              tag_v<accel_type>, value.individual_settings);

          auto &intersector = to_bulk_.get(tag_v<accel_type>);
          intersector.set_settings_intersectable(
              value.to_bulk_settings, intersectable_scene.intersector);

          // Specify exact type to make sure we get a reference for
          // intersector.
          return std::tuple<decltype(intersectable_scene.scene),
                            decltype(intersector)>{intersectable_scene.scene,
                                                   intersector};
        });
      }
    }();

    auto scene = std::get<0>(intersectable_items);
    auto &intersector = std::get<1>(intersectable_items);

    auto light_sampler =
        light_samplers_.get(tag_v<light_sampler_type>)
            .gen(settings.light_sampler.get(tag_v<light_sampler_type>),
                 s.emissive_clusters(), s.emissive_cluster_ends_per_mesh(),
                 s.materials().as_unsized(), s.transformed_mesh_objects(),
                 s.transformed_mesh_idxs(), s.triangles().as_unsized());

    auto dir_sampler =
        dir_samplers_.get(tag_v<dir_sampler_type>)
            .gen(settings.dir_sampler.get(tag_v<dir_sampler_type>));

    auto term_prob = term_probs_.get(tag_v<term_prob_type>)
                         .gen(settings.term_prob.get(tag_v<term_prob_type>));

    unsigned n_locations = x_dim * y_dim;

    auto rng =
        rngs_.get(tag_v<rng_type>)
            .gen(settings.rng.get(tag_v<rng_type>), samples_per, n_locations);

    // not really any good way to infer these arguments...
    using Components = integrate::RenderingEquationComponents<
        decltype(scene), decltype(light_sampler), decltype(dir_sampler),
        decltype(term_prob)>;
    using Items = integrate_image::Items<Components, decltype(rng)>;
    integrate_image::Inputs<Items> inputs = {
        .items =
            {
                .components =
                    {
                        .scene = scene,
                        .light_sampler = light_sampler,
                        .dir_sampler = dir_sampler,
                        .term_prob = term_prob,
                    },
                .rng = rng,
                .film_to_world = s.film_to_world(),
            },
        .output_as_bgra_32 = output_as_bgra_32,
        .bgra_32 = bgra_32_,
        .samples_per = samples_per,
        .x_dim = x_dim,
        .y_dim = y_dim,
        .show_progress = show_progress,
    };
    return [&]() -> ExecVecT<FloatRGB> * {
      if constexpr (kernel_approach == KernelApproach::MegaKernel) {
        return integrate_image::mega_kernel::Run<exec>::run(
            inputs, intersector, kernel_approach_settings, float_rgb_,
            reduced_float_rgb_);
      } else {
        static_assert(kernel_approach == KernelApproach::Streaming);

        auto &streaming_state = streaming_state_.get(
            tag_v<make_meta_tuple(light_sampler_type, rng_type)>);

        integrate_image::streaming::Run<exec>::run(
            inputs, intersector, streaming_state, kernel_approach_settings,
            float_rgb_);

        return &float_rgb_;
      }
    }();
  });

  // We could save a copy in the cpu case by directly outputing to the input
  // bgra_32/float_rgb when possible.
  // However, the perf gains are small and we don't really
  // case about the performance of the cpu use case.
  if (output_as_bgra_32) {
    thrust::copy(bgra_32_.begin(), bgra_32_.end(), bgra_32_output.begin());
  } else {
    thrust::copy(float_rgb_out->begin(), float_rgb_out->end(),
                 float_rgb_output.begin());
  }
}
} // namespace render
