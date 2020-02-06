#include "lib/timer.h"
#include "ray/detail/accel/aabb.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
DirTreeLookup DirTreeGeneratorImpl<execution_model>::generate(
    const Eigen::Projective3f &world_to_film,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights, const Eigen::Vector3f &min_bound,
    const Eigen::Vector3f &max_bound) {
  is_x_ = true;

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

  current_edges_->resize_all(double_aabb_size);
  other_edges_->resize_all(double_aabb_size);
  sorted_by_z_min_.first->resize_all(aabbs_.size());
  sorted_by_z_max_.first->resize_all(aabbs_.size());

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

  dir_trees_.resize(transforms_.size());

  for (unsigned i = 0; i < transforms_.size(); i++) {
    Span<const DirTreeNode> nodes = nodes_;
    Span<const DirTreeNode> nodes_shifted(nodes.data() + i, nodes_.size() - i);
    dir_trees_[i] = DirTree(transforms_[i], nodes_shifted, actions_);
  }

  return DirTreeLookup(dir_trees_, 0, 1, lights.size() + 1, partition);
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
