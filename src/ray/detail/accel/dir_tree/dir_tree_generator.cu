#include "lib/timer.h"
#include "ray/detail/accel/aabb.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "scene/camera.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <future>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {

#if 0
struct CreateTraversal : public thrust::binary_function<int, int, Traversal> {
  HOST_DEVICE Traversal operator()(int start, int end) {
    return Traversal(start, end);
  }
};
#endif

template <ExecutionModel execution_model>
DirTreeLookup DirTreeGenerator<execution_model>::get_dir_trees(
    const Eigen::Projective3f &world_to_film,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights, const Eigen::Vector3f &min_bound,
    const Eigen::Vector3f &max_bound) {
  Timer setup_dir_tree_transforms_timer;

  unsigned region_target = 32;

  HalfSpherePartition partition(region_target, sphere_partition_regions_);

  unsigned num_dir_trees = 1 + lights.size() + partition.size();

  transforms_.resize(num_dir_trees);
  sort_offsets_.resize(num_dir_trees);

  unsigned transform_idx = 0;

  Eigen::Vector3f overall_min = Eigen::Vector3f::Zero();
  Eigen::Vector3f overall_max = Eigen::Vector3f::Zero();

  auto add_transform = [&](const Eigen::Projective3f &transf) {
    transforms_[transform_idx] = transf;

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

  setup_dir_tree_transforms_timer.report("setup dir tree transforms");

  Timer project_shapes_dir_tree_timer;

  bounds_.resize(shapes.size());

  std::transform(shapes.begin(), shapes.end(), bounds_.begin(),
                 [](const scene::ShapeData &shape) {
                   return get_bounding(shape.get_transform());
                 });

  // TODO: batching...
  aabbs_.resize(bounds_.size() * transforms_.size());

  compute_aabbs();

  project_shapes_dir_tree_timer.report("project shapes dir tree");

  Timer copy_to_sortable_timer;

  unsigned double_aabb_size = aabbs_.size() * 2;

  sorted_by_x_edges_.resize(double_aabb_size);
  sorted_by_y_edges_.resize(double_aabb_size);
  sorted_by_z_min_.resize(aabbs_.size());
  sorted_by_z_max_.resize(aabbs_.size());

  sorting_values_[0].resize(double_aabb_size);
  sorting_values_[1].resize(double_aabb_size);
  sorting_values_[2].resize(aabbs_.size());
  sorting_values_[3].resize(aabbs_.size());

  indexes_[0].resize(double_aabb_size);
  indexes_[1].resize(double_aabb_size);
  indexes_[2].resize(aabbs_.size());
  indexes_[3].resize(aabbs_.size());

  while (thrust_data_.size() < 4) {
    thrust_data_.push_back(ThrustData<execution_model>());
  }

  copy_to_sortable(shapes.size());

  copy_to_sortable_timer.report("copy to sortable");

  Timer fill_indexes_timer;

  fill_indexes();

  fill_indexes_timer.report("fill indexes");

  Timer sort_timer;

  sort();

  sort_timer.report("sort");

  Timer permute_timer;

  permute();

  permute_timer.report("permute");
}

template class DirTreeGenerator<ExecutionModel::CPU>;
template class DirTreeGenerator<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
