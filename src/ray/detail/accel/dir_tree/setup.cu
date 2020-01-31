#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "ray/detail/accel/dir_tree/impl/sphere_partition_impl.h"
#include "scene/camera.h"
#include "lib/async_for.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
HalfSpherePartition DirTreeGenerator<execution_model>::setup(
    const Eigen::Projective3f &world_to_film,
    SpanSized<const scene::Light> lights, const Eigen::Vector3f &min_bound,
    const Eigen::Vector3f &max_bound) {
  unsigned region_target = 32;

  HalfSpherePartition partition(region_target, sphere_partition_regions_);

  unsigned num_dir_trees = 1 + lights.size() + partition.size();

  transforms_.resize(num_dir_trees);
  sort_offsets_.resize(num_dir_trees);
  groups_.resize_all(num_dir_trees);
  groups_cpu_.resize_all(num_dir_trees);

  unsigned transform_idx = 0;

  Eigen::Vector3f overall_min = Eigen::Vector3f::Zero();
  Eigen::Vector3f overall_max = Eigen::Vector3f::Zero();

  auto add_transform = [&](const Eigen::Projective3f &transf) {
    transforms_[transform_idx] = transf;

    unsigned end_edges = (transform_idx + 1) * num_shapes_ * 2;
    unsigned end_z = (transform_idx + 1) * num_shapes_;

    groups_cpu_[0][transform_idx] = end_edges;
    groups_cpu_[1][transform_idx] = end_edges;
    groups_cpu_[2][transform_idx] = end_z;

    auto [transf_min_bound, transf_max_bound] =
        get_transformed_bounds(transf, min_bound, max_bound);

    for (auto axis : {0, 1, 2}) {
      if (std::abs(overall_min[axis]) < std::abs(overall_max[axis])) {
        sort_offsets_[transform_idx][axis] =
            overall_min[axis] - transf_max_bound[axis];
        overall_min[axis] =
            transf_min_bound[axis] + sort_offsets_[transform_idx][axis];
      } else {
        sort_offsets_[transform_idx][axis] =
            overall_max[axis] - transf_min_bound[axis];
        overall_max[axis] =
            transf_max_bound[axis] + sort_offsets_[transform_idx][axis];
      }
    }

    transform_idx++;
  };

  // camera transform...
  add_transform(world_to_film);

  auto add_transform_vec = [&](bool is_loc, const Eigen::Vector3f &loc_or_dir) {
    if (is_loc) {
      add_transform(scene::get_unhinging(30.0f) *
                    Eigen::Translation3f(-loc_or_dir));
    } else {
      add_transform(Eigen::Projective3f(
          scene::look_at(loc_or_dir, Eigen::Vector3f(0, 1, 0))));
    }
  };

  for (const auto &light : lights) {
    light.visit([&](auto &&light_data) {
      using T = std::decay_t<decltype(light_data)>;
      if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
        add_transform_vec(false, light_data.direction);
      } else {
        add_transform_vec(true, light_data.position);
      }
    });
  }

  for (unsigned collar = 0; collar < partition.regions().size(); collar++) {
    unsigned start = partition.regions()[collar].start_index;
    unsigned end = partition.regions()[collar].end_index;
    for (unsigned region = 0; region < end - start; region++) {
      add_transform_vec(false, partition.get_center_vec(collar, region));
    }
  }

  async_for<true>(0, 3, [&](unsigned axis) {
    thrust::copy(thrust_data_[axis].execution_policy(),
                 groups_cpu_[axis].begin(), groups_cpu_[axis].end(),
                 groups_[axis].begin());
  });

  return partition;
}

template class DirTreeGenerator<ExecutionModel::CPU>;
template class DirTreeGenerator<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
