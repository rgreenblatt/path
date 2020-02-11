#include "lib/async_for.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/impl/sphere_partition_impl.h"
#include "scene/camera.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
HalfSpherePartition DirTreeGeneratorImpl<execution_model>::setup(
    const Eigen::Projective3f &world_to_film,
    SpanSized<const scene::Light> lights, const Eigen::Vector3f &min_bound,
    const Eigen::Vector3f &max_bound) {
  /* unsigned region_target = 32; */
  unsigned region_target = 1;

  HalfSpherePartition partition(region_target, sphere_partition_regions_);

  unsigned num_dir_trees = 1 + lights.size() + partition.size();

  transforms_.resize(num_dir_trees);
  sort_offsets_.resize(num_dir_trees);
  axis_groups_.first->resize_all(num_dir_trees);
  axis_groups_cpu_.resize_all(num_dir_trees);
  open_mins_before_group_.first->resize(num_dir_trees);
  thrust::fill(thrust_data_[0].execution_policy(),
               open_mins_before_group_.first->data(),
               open_mins_before_group_.first->data() + num_dir_trees, 0);
  num_per_group_.first->resize(num_dir_trees);
  thrust::fill(thrust_data_[0].execution_policy(), num_per_group_.first->data(),
               num_per_group_.first->data() + num_dir_trees, num_shapes_);

  unsigned transform_idx = 0;

  Eigen::Vector3f overall_max = Eigen::Vector3f::Zero();

  auto add_transform = [&](const Eigen::Projective3f &transf) {
    transforms_[transform_idx] = transf;

    unsigned end_edges = (transform_idx + 1) * num_shapes_ * 2;
    unsigned end_z = (transform_idx + 1) * num_shapes_;

    axis_groups_cpu_[0][transform_idx] = end_edges;
    axis_groups_cpu_[1][transform_idx] = end_edges;
    axis_groups_cpu_[2][transform_idx] = end_z;

    // line up ranges end to end
    auto [transf_min_bound, transf_max_bound] =
        get_transformed_bounds(transf, min_bound, max_bound);

    for (auto axis : {0, 1, 2}) {
      if (transform_idx == 0) {
        sort_offsets_[transform_idx][axis] = 0;
      } else {
        sort_offsets_[transform_idx][axis] =
            overall_max[axis] - transf_min_bound[axis];
      }
      overall_max[axis] =
          transf_max_bound[axis] + sort_offsets_[transform_idx][axis] + 1e-5;
    }

#ifdef DEBUG_PRINT
    std::cout << "transform:\n" << transf << std::endl;
    std::cout << "transformed min bound:\n" << transf_min_bound << std::endl;
    std::cout << "transformed max bound:\n" << transf_max_bound << std::endl;
    std::cout << "offset:\n" << sort_offsets_[transform_idx] << std::endl;
#endif

    transform_idx++;
  };

  // camera transform...
  add_transform(world_to_film);

  auto add_transform_vec = [&](bool is_loc, const Eigen::Vector3f &loc_or_dir) {
    if (is_loc) {
      add_transform(scene::get_unhinging(30.0f) *
                    Eigen::Translation3f(-loc_or_dir));
    } else {
#pragma message("This can fail if direction is too close to look")
      const auto t = Eigen::Projective3f(
          scene::look_at(loc_or_dir, Eigen::Vector3f(1, 1, 0)));
#ifdef DEBUG_PRINT
      std::cout << "loc or dir: " << loc_or_dir << std::endl;
      std::cout << "transform: " << t << std::endl;
#endif
      add_transform(t);
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

  for (unsigned colatitude_div_idx = 0;
       colatitude_div_idx < partition.colatitude_divs().size();
       colatitude_div_idx++) {
    unsigned start =
        partition.colatitude_divs()[colatitude_div_idx].start_index;
    unsigned end = partition.colatitude_divs()[colatitude_div_idx].end_index;
    for (unsigned region = 0; region < end - start; region++) {
      add_transform_vec(false,
                        partition.get_center_vec(colatitude_div_idx, region));
    }
  }

  for (uint8_t axis = 0; axis < 3; axis++) {
    thrust::copy(thrust_data_[axis].execution_policy(),
                 axis_groups_cpu_[axis].begin(), axis_groups_cpu_[axis].end(),
                 axis_groups_.first.get()[axis].begin());
  }

  return partition;
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
