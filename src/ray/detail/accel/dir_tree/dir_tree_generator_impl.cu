#include "lib/timer.h"
#include "ray/detail/accel/aabb.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
DirTreeGeneratorImpl<execution_model>::DirTreeGeneratorImpl()
    : use_async_(execution_model != ExecutionModel::CPU), is_x_(true),
      current_edges_(edges_underlying_[0]), other_edges_(edges_underlying_[1]),
      other_edges_new_(edges_underlying_[2]),
      sorted_by_z_min_(sorted_by_z_min_underlying_.first,
                       sorted_by_z_min_underlying_.second),
      sorted_by_z_max_(sorted_by_z_max_underlying_.first,
                       sorted_by_z_max_underlying_.second),
      current_edges_min_max_(group_min_max_underlying_[0]),
      other_edges_min_max_(group_min_max_underlying_[1]),
      current_edges_min_max_new_(group_min_max_underlying_[2]),
      other_edges_min_max_new_(group_min_max_underlying_[3]),
      current_edges_keys_(x_edges_keys_), other_edges_keys_(y_edges_keys_),
      axis_groups_(axis_groups_underlying_.first,
                   axis_groups_underlying_.second),
      open_mins_before_group_(open_mins_before_group_underlying_.first,
                              open_mins_before_group_underlying_.second),
      num_per_group_(num_per_group_underlying_.first,
                     num_per_group_underlying_.second),
      better_than_no_split_(better_than_no_split_underlying_.first,
                            better_than_no_split_underlying_.second),
      node_offset_(0), output_values_offset_(0) {}

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

#ifdef DEBUG_PRINT
  if constexpr (execution_model == ExecutionModel::CPU) {
    std::cout << "num shapes: " << shapes.size() << std::endl;
    std::cout << "bounds: " << bounds_ << std::endl;
  }
#endif

  // TODO: batching...
  aabbs_.resize(bounds_.size() * transforms_.size());

  compute_aabbs();

#ifdef DEBUG_PRINT
  if constexpr (execution_model == ExecutionModel::CPU) {
    std::cout << "num transforms: " << transforms_.size() << std::endl;
    std::cout << transforms_ << std::endl;
    std::cout << "aabbs: " << aabbs_ << std::endl;
    std::cout << "offsets: " << sort_offsets_ << std::endl;
  }
#endif

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

  auto debug_print_sorting = [&](const std::string &s) {
#ifdef DEBUG_PRINT
    if constexpr (execution_model == ExecutionModel::CPU) {
      std::cout << s << ": " << std::endl;
      std::cout << "sorting values x: " << sorting_values_[0] << std::endl;
      std::cout << "sorting values y: " << sorting_values_[1] << std::endl;
      std::cout << "sorting values z max: " << sorting_values_[2] << std::endl;
      std::cout << "sorting values z max: " << sorting_values_[3] << std::endl;
      std::cout << "idxs x: " << indexes_[0] << std::endl;
      std::cout << "idxs y: " << indexes_[1] << std::endl;
      std::cout << "idxs z max: " << indexes_[2] << std::endl;
      std::cout << "idxs z max: " << indexes_[3] << std::endl;
    }
#endif
  };

  debug_print_sorting("before sort");

  fill_indexes_timer.report("fill indexes");

  Timer sort_timer;

  sort();

  sort_timer.report("sort");

  debug_print_sorting("after sort");

  Timer permute_timer;

  permute();

  permute_timer.report("permute");

#ifdef DEBUG_PRINT
  if constexpr (execution_model == ExecutionModel::CPU) {
    {
      std::cout << "x groups: " << axis_groups_.first.get()[0] << std::endl;
      std::cout << "y groups: " << axis_groups_.first.get()[1] << std::endl;
      std::cout << "z groups: " << axis_groups_.first.get()[2] << std::endl;
    }
    {
      std::cout << "x other mins: " << current_edges_->other_mins()
                << std::endl;
      std::cout << "x other maxs: " << current_edges_->other_maxs()
                << std::endl;
      std::cout << "x values: " << current_edges_->values() << std::endl;
      std::cout << "x is mins: " << current_edges_->is_mins() << std::endl;
      std::cout << "x group min max: " << current_edges_min_max_.get()
                << std::endl;
    }
    {
      std::cout << "y other mins: " << other_edges_->other_mins() << std::endl;
      std::cout << "y other maxs: " << other_edges_->other_maxs() << std::endl;
      std::cout << "y values: " << other_edges_->values() << std::endl;
      std::cout << "y is mins: " << other_edges_->is_mins() << std::endl;
      std::cout << "y group min max: " << other_edges_min_max_.get()
                << std::endl;
    }
    {
      std::cout << "sorted z min x mins: " << sorted_by_z_min_.first->x_mins()
                << std::endl;
      std::cout << "sorted z min x maxs: " << sorted_by_z_min_.first->x_maxs()
                << std::endl;
      std::cout << "sorted z min y mins: " << sorted_by_z_min_.first->y_mins()
                << std::endl;
      std::cout << "sorted z min y maxs: " << sorted_by_z_min_.first->y_maxs()
                << std::endl;
      std::cout << "sorted z min z mins: " << sorted_by_z_min_.first->z_mins()
                << std::endl;
      std::cout << "sorted z min z maxs: " << sorted_by_z_min_.first->z_maxs()
                << std::endl;
      std::cout << "sorted z min idxs: " << sorted_by_z_min_.first->idxs()
                << std::endl;
    }
    {
      std::cout << "sorted z max x mins: " << sorted_by_z_max_.first->x_mins()
                << std::endl;
      std::cout << "sorted z max x maxs: " << sorted_by_z_max_.first->x_maxs()
                << std::endl;
      std::cout << "sorted z max y mins: " << sorted_by_z_max_.first->y_mins()
                << std::endl;
      std::cout << "sorted z max y maxs: " << sorted_by_z_max_.first->y_maxs()
                << std::endl;
      std::cout << "sorted z max z maxs: " << sorted_by_z_max_.first->z_maxs()
                << std::endl;
      std::cout << "sorted z max z maxs: " << sorted_by_z_max_.first->z_maxs()
                << std::endl;
      std::cout << "sorted z max idxs: " << sorted_by_z_max_.first->idxs()
                << std::endl;
    }
  }
#endif

  // set void node as node at zero
  nodes_[0] = DirTreeNode(0, 0);

  Timer construct_trees_timer;

  construct();

  construct_trees_timer.report("construct trees");

  dir_trees_.resize(transforms_.size());

  for (unsigned i = 0; i < transforms_.size(); i++) {
    Span<const DirTreeNode> nodes = nodes_;
    Span<const DirTreeNode> nodes_shifted(nodes.data() + i, nodes_.size() - i);
    /* dir_trees_[i] = DirTree(transforms_[i], nodes_shifted, actions_); */
  }

  return DirTreeLookup(dir_trees_, 0, 1, lights.size() + 1, partition);
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
