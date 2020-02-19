#pragma once

#include "render/detail/renderer_impl.h"
namespace render {
namespace detail {

template <ExecutionModel execution_model>
void dispatch_compute_intensities(const PerfSettings &settings) {
  dispatch_value(
      [&](auto &&settings_tup) {
        constexpr CompileTimePerfSettings compile_time_settings =
            std::decay_t<decltype(settings_tup)>::value;
    /* constexpr auto tri_accel_type = settings. */
#if 0
        constexpr auto tri_accel_type = std::decay_t<decltype(i)>::value;
        using AcceleratorType = intersect::accel::AcceleratorType;

        auto &v = stored_mesh_accels_.template get_item<tri_accel_type>();

        using TriRefType = typename std::decay_t<decltype(v)>::RefType;
        using TriSettings = typename std::decay_t<decltype(v)>::Settings;

        TriSettings triangle_accel_settings;
        if constexpr (tri_accel_type == AcceleratorType::DirTree) {
          triangle_accel_settings.s_a_heuristic_settings = {
              dir_tree_triangle_traversal_cost,
              dir_tree_triangle_intersection_cost};
          triangle_accel_settings.num_dir_trees = num_dir_trees_triangle;
        } else if constexpr (tri_accel_type == AcceleratorType::KDTree) {
          triangle_accel_settings.s_a_heuristic_settings = {
              kd_tree_triangle_traversal_cost,
              kd_tree_triangle_intersection_cost};
        }

        unsigned num_meshs = s.mesh_paths().size();

        v.reset();

        std::vector<TriRefType> cpu_refs(num_meshs);
        std::vector<uint8_t> ref_set(num_meshs, 0);

        for (unsigned i = 0; i < num_meshs; i++) {
          auto ref_op = v.query(s.mesh_paths()[i]);
          if (ref_op.has_value()) {
            cpu_refs[i] = *ref_op;
            ref_set[i] = true;
          }
        }

        assert(s.mesh_aabbs().size() == num_meshs);

        for (unsigned i = 0; i < num_meshs; i++) {
          if (!ref_set[i]) {
            const auto &aabb = s.mesh_aabbs()[i];
            cpu_refs[i] = v.add(s.triangles(), get_previous(i, s.mesh_ends()),
                                s.mesh_ends()[i], aabb.get_min_bound(),
                                aabb.get_max_bound());
          }
        }

        ExecVecT<TriRefType> refs(cpu_refs.begin(), cpu_refs.end());

        intersect::accel::run_over_accelerator_types(
            [&](auto &&i) {
              constexpr auto mesh_accel_type = std::decay_t<decltype(i)>::value;

              using MeshInstanceRef =
                  intersect::accel::MeshInstanceRef<TriRefType>;
              using MeshGenerator =
                  intersect::accel::Generator<MeshInstanceRef, execution_model,
                                              mesh_accel_type>;
              using MeshSettings = intersect::accel::Settings<mesh_accel_type>;

              MeshSettings mesh_accel_settings;
              if constexpr (mesh_accel_type == AcceleratorType::DirTree) {
                mesh_accel_settings.s_a_heuristic_settings = {
                    dir_tree_mesh_traversal_cost,
                    dir_tree_mesh_intersection_cost};
                mesh_accel_settings.num_dir_trees = num_dir_trees_mesh;
              } else if constexpr (mesh_accel_type == AcceleratorType::KDTree) {
                mesh_accel_settings.s_a_heuristic_settings = {
                    kd_tree_mesh_traversal_cost,
                    kd_tree_mesh_intersection_cost};
              }

              MeshGenerator generator;

              unsigned num_mesh_instances = s.mesh_instances().size();

              std::vector<MeshInstanceRef> instance_refs(num_mesh_instances);

              for (unsigned i = 0; i < num_mesh_instances; ++i) {
                instance_refs[i] =
                    s.mesh_instances()[i].get_ref(Span<const TriRefType>{refs});
              }

              const auto &aabb = s.overall_aabb();

              auto mesh_instance_accel_ref = generator.gen(
                  instance_refs, 0, num_mesh_instances, aabb.get_min_bound(),
                  aabb.get_max_bound(), mesh_accel_settings);

              compute_intensities<execution_model>(
                  division, samples_per, x_dim, y_dim, block_size,
                  mesh_instance_accel_ref, intermediate_intensities_,
                  final_intensities_, triangle_data, materials,
                  s.film_to_world());
            },
            mesh_accel_type);
#endif
      },
      settings.compile_time.values());
}
} // namespace detail
} // namespace render
