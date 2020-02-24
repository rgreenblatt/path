#pragma once

#include "compile_time_dispatch/dispatch_value.h"
#include "intersect/accel/loop_all.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "lib/group.h"
#include "render/detail/compute_intensities.h"
#include "render/detail/divide_work.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/tone_map.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(Span<BGRA> pixels,
                                           const scene::Scene &s,
                                           unsigned samples_per, unsigned x_dim,
                                           unsigned y_dim,
                                           const Settings &settings, bool) {
  if (samples_per > std::numeric_limits<uint16_t>::max()) {
    std::cerr << "more samples than allowed" << std::endl;
    return;
  }

  unsigned block_size = 512;
  unsigned target_work_per_thread = 4;

  auto division = divide_work(samples_per, x_dim, y_dim, block_size,
                              target_work_per_thread);

  Span<const scene::TriangleData> triangle_data;
  Span<const material::Material> materials;
  Span<BGRA> output_pixels;

  if constexpr (execution_model == ExecutionModel::GPU) {
    auto inp_t_data = s.triangle_data();
    auto inp_materials = s.materials();

    triangle_data_.resize(inp_t_data.size());
    materials_.resize(inp_materials.size());

    thrust::copy(inp_t_data.begin(), inp_t_data.end(), triangle_data_.begin());
    thrust::copy(inp_materials.begin(), inp_materials.end(),
                 materials_.begin());

    triangle_data = triangle_data_;
    materials = materials_;

    if (division.num_sample_blocks != 1) {
      intensities_.resize(division.num_sample_blocks * x_dim * y_dim);
    }

    bgra_.resize(x_dim * y_dim);
    output_pixels = bgra_;
  } else {
    triangle_data = s.triangle_data();
    materials = s.materials();
    output_pixels = pixels;
  }

  dispatch_value(
      [&](auto &&settings_tup) {
        // TODO: consider dispatching more generically...
        constexpr CompileTimeSettings compile_time_settings =
            std::decay_t<decltype(settings_tup)>::value;

        constexpr auto triangle_accel_type =
            compile_time_settings.triangle_accel_type();

        auto &triangle_accels =
            stored_triangle_accels_.template get_item<triangle_accel_type>();

        using Triangle = intersect::Triangle;
        using TriAccel = intersect::accel::AccelT<triangle_accel_type,
                                                  execution_model, Triangle>;
        using TriRefType = typename TriAccel::Ref;

        unsigned num_meshs = s.mesh_paths().size();

        triangle_accels.reset();

        std::vector<TriRefType> cpu_refs(num_meshs);
        std::vector<uint8_t> ref_set(num_meshs, 0);

        auto specific_settings =
            settings.triangle_accel.template get_item<triangle_accel_type>();

        for (unsigned i = 0; i < num_meshs; i++) {
          auto ref_op =
              triangle_accels.query(s.mesh_paths()[i], specific_settings);
          if (ref_op.has_value()) {
            cpu_refs[i] = *ref_op;
            ref_set[i] = true;
          }
        }

        assert(s.mesh_aabbs().size() == num_meshs);

        for (unsigned i = 0; i < num_meshs; i++) {
          if (!ref_set[i]) {
            const auto &aabb = s.mesh_aabbs()[i];
            cpu_refs[i] = triangle_accels.add(
                s.triangles(), get_previous(i, s.mesh_ends()), s.mesh_ends()[i],
                aabb, specific_settings, s.mesh_paths()[i]);
          }
        }

        ExecVecT<TriRefType> refs(cpu_refs.begin(), cpu_refs.end());

        constexpr auto mesh_accel_type =
            compile_time_settings.mesh_accel_type();

        using MeshInstanceRef = intersect::MeshInstanceRef<TriRefType>;

        using MeshAccel =
            intersect::accel::AccelT<mesh_accel_type, execution_model,
                                     MeshInstanceRef>;

        MeshAccel mesh_accel;

        unsigned num_mesh_instances = s.mesh_instances().size();

        std::vector<MeshInstanceRef> instance_refs(num_mesh_instances);

        for (unsigned i = 0; i < num_mesh_instances; ++i) {
          instance_refs[i] =
              s.mesh_instances()[i].get_ref(Span<const TriRefType>{refs});
        }

        const auto &aabb = s.overall_aabb();

        auto mesh_instance_accel_ref = mesh_accel.gen(
            settings.mesh_accel.template get_item<mesh_accel_type>(),
            instance_refs, 0, num_mesh_instances, aabb);

        constexpr auto light_sampler_type =
            compile_time_settings.light_sampler_type();
        constexpr auto dir_sampler_type =
            compile_time_settings.dir_sampler_type();
        constexpr auto term_prob_type = compile_time_settings.term_prob_type();
        constexpr auto rng_type = compile_time_settings.rng_type();

        auto light_sampler =
            light_samplers_.template get_item<light_sampler_type>().gen(
                settings.light_sampler.template get_item<light_sampler_type>(),
                s.emissive_groups(), s.emissive_group_ends_per_mesh(),
                s.materials(), s.mesh_instances());

        auto dir_sampler =
            dir_samplers_.template get_item<dir_sampler_type>().gen(
                settings.dir_sampler.template get_item<dir_sampler_type>());

        auto term_prob = term_probs_.template get_item<term_prob_type>().gen(
            settings.term_prob.template get_item<term_prob_type>());

        // TODO
        const unsigned max_draws_per_sample = 100;

        auto rng = rngs_.template get_item<rng_type>().gen(
            settings.rng.template get_item<rng_type>(), samples_per, x_dim,
            y_dim, max_draws_per_sample);

        compute_intensities(division, samples_per, x_dim, y_dim, block_size,
                            mesh_instance_accel_ref, light_sampler, dir_sampler,
                            term_prob, rng, output_pixels, intensities_,
                            triangle_data, materials, s.film_to_world());
      },
      settings.compile_time.values());

  if constexpr (execution_model == ExecutionModel::GPU) {
    if (division.num_sample_blocks != 1) {
      tone_map<execution_model>(intensities_, bgra_);
    }

    thrust::copy(bgra_.begin(), bgra_.end(), pixels.begin());
  }
}
} // namespace detail
} // namespace render
