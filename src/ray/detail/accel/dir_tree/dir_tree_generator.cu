#include "lib/timer.h"
#include "ray/detail/accel/aabb.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

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
DirTreeLookup DirTreeGenerator<execution_model>::generate(
    const Eigen::Projective3f &world_to_film,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights, const Eigen::Vector3f &min_bound,
    const Eigen::Vector3f &max_bound) {
  num_shapes_ = shapes.size();

  Timer setup_timer;

  auto partition = setup(world_to_film, lights, min_bound, max_bound);

  setup_timer.report("setup");

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

  x_edges_.resize_all(double_aabb_size);
  y_edges_.resize_all(double_aabb_size);
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

  copy_to_sortable();

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

  Timer construct_trees_timer;

  construct();

  construct_trees_timer.report("construct trees");

  return DirTreeLookup(Span<const DirTree>(0, 0), 0, 0, 0, partition);
}

template class DirTreeGenerator<ExecutionModel::CPU>;
template class DirTreeGenerator<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
