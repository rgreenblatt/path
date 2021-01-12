#pragma once

#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/triangle_impl.h"
#include "intersectable_scene/to_bulk_impl.h"
#include "meta/dispatch_value.h"
#include "render/detail/integrate_image.h"
#include "render/detail/reduce_intensities_gpu.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/settings_compile_time_impl.h"
#include "work_division/work_division.h"

namespace render {
using namespace detail;

template <ExecutionModel exec>
void Renderer::Impl<exec>::general_render(
    bool output_as_bgra, Span<BGRA> pixels, Span<Eigen::Array3f> intensities,
    const scene::Scene &s, unsigned samples_per, unsigned x_dim, unsigned y_dim,
    const Settings &settings, bool show_progress, bool) {
  WorkDivision division = WorkDivision(
      settings.general_settings.computation_settings.render_work_division,
      samples_per, x_dim, y_dim);

  Span<BGRA> output_pixels;
  Span<Eigen::Array3f> output_intensities;

  if (output_as_bgra) {
    if constexpr (exec == ExecutionModel::GPU) {
      if (division.num_sample_blocks() != 1 || output_as_bgra) {
        intensities_.resize(division.num_sample_blocks() * x_dim * y_dim);
        output_intensities = intensities_;
      }

      bgra_.resize(x_dim * y_dim);
      output_pixels = bgra_;
    } else {
      output_pixels = pixels;
    }
  } else {
    if constexpr (exec == ExecutionModel::GPU) {
      intensities_.resize(division.num_sample_blocks() * x_dim * y_dim);
      output_intensities = intensities_;
    } else {
      output_intensities = intensities;
    }
  }

  dispatch_value(
      [&](auto compile_time_holder) {
        constexpr auto compile_time = decltype(compile_time_holder)::value;

        constexpr auto intersection_type = compile_time.intersection_type;
        constexpr auto light_sampler_type = compile_time.light_sampler_type;
        constexpr auto dir_sampler_type = compile_time.dir_sampler_type;
        constexpr auto term_prob_type = compile_time.term_prob_type;
        constexpr auto rng_type = compile_time.rng_type;

        // this will need to change somewhat... when another value is added...
        static_assert(AllValues<IntersectionApproach>.size() == 2);
        constexpr auto intersection_approach = intersection_type.type();
        constexpr auto accel_type =
            intersection_type.get(TAG(intersection_approach));

        const auto &all_intersection_settings =
            settings.intersection.get(TAG(intersection_approach));
        const auto &accel_settings = [&]() -> const auto & {
          if constexpr (intersection_approach ==
                        IntersectionApproach::MegaKernel) {
            return all_intersection_settings;
          } else if constexpr (intersection_approach ==
                               IntersectionApproach::StreamingFromGeneral) {
            return all_intersection_settings.accel;
          }
        }
        ().get(TAG(accel_type));

        auto intersectable_scene = stored_scene_generators_.get(TAG(accel_type))
                                       .gen({accel_settings}, s);

        auto &intersector = [&]() -> auto & {
          if constexpr (intersection_approach ==
                        IntersectionApproach::MegaKernel) {
            return intersectable_scene.intersector;
          } else if constexpr (intersection_approach ==
                               IntersectionApproach::StreamingFromGeneral) {
            auto &out = to_bulk_.get(TAG(accel_type));
            out.set_settings_intersectable(
                all_intersection_settings.to_bulk_settings,
                intersectable_scene.intersector);

            return out;
          }
        }
        ();

        auto light_sampler =
            light_samplers_.get(TAG(light_sampler_type))
                .gen(settings.light_sampler.get(TAG(light_sampler_type)),
                     s.emissive_clusters(), s.emissive_cluster_ends_per_mesh(),
                     s.materials().as_unsized(), s.transformed_mesh_objects(),
                     s.transformed_mesh_idxs(), s.triangles().as_unsized());

        auto dir_sampler =
            dir_samplers_.get(TAG(dir_sampler_type))
                .gen(settings.dir_sampler.get(TAG(dir_sampler_type)));

        auto term_prob = term_probs_.get(TAG(term_prob_type))
                             .gen(settings.term_prob.get(TAG(term_prob_type)));

        unsigned n_locations = x_dim * y_dim;

        auto rng =
            rngs_.get(TAG(rng_type))
                .gen(settings.rng.get(TAG(rng_type)), samples_per, n_locations);

#if 1
        // TODO: won't be needed when P1021R4
        // (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1021r4.html)
        // is implemented
        using Components = integrate::RenderingEquationComponents<
            decltype(intersectable_scene.scene), decltype(light_sampler),
            decltype(dir_sampler), decltype(term_prob)>;
        using Items = IntegrateImageItems<Components, decltype(rng)>;
        using Inp =
            IntegrateImageInputs<Items, std::decay_t<decltype(intersector)>>;

        IntegrateImage<exec>::run(Inp{
            .items =
                {
                    .base =
                        {
                            .output_as_bgra = output_as_bgra,
                            .samples_per = samples_per,
                            .pixels = output_pixels,
                            .intensities = output_intensities,
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
        });
#endif
      },
      settings.compile_time());

  if constexpr (exec == ExecutionModel::GPU) {
    auto intensities_gpu = reduce_intensities_gpu(
        output_as_bgra, division.num_sample_blocks(), samples_per,
        &intensities_, &reduced_intensities_, bgra_);
    always_assert(intensities_gpu != nullptr);
    always_assert(intensities_gpu == &intensities_ ||
                  intensities_gpu == &reduced_intensities_);

    if (output_as_bgra) {
      thrust::copy(bgra_.begin(), bgra_.end(), pixels.begin());
    } else {
      always_assert(intensities_gpu != nullptr);
      always_assert(intensities_gpu == &intensities_ ||
                    intensities_gpu == &reduced_intensities_);
      thrust::copy(intensities_gpu->begin(), intensities_gpu->end(),
                   intensities.begin());
    }
  }
}
} // namespace render
