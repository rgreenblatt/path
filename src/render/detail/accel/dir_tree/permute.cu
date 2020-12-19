#include "lib/async_for.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

#include <thrust/iterator/permutation_iterator.h>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::permute() {
  unsigned num_dir_trees = num_groups();
  current_edges_min_max_->resize(num_dir_trees);
  other_edges_min_max_->resize(num_dir_trees);

  nodes_.resize(4 * num_dir_trees + 1);
  node_offset_ = nodes_.size();

  // set void node as node at zero
  nodes_[0] = DirTreeNode(0, 0);

  Span<DirTreeNode> nodes = nodes_;

  min_x_tree_.resize(num_dir_trees);
  min_y_tree_.resize(num_dir_trees);
  min_z_tree_.resize(num_dir_trees);
  max_x_tree_.resize(num_dir_trees);
  max_y_tree_.resize(num_dir_trees);
  max_z_tree_.resize(num_dir_trees);

  async_for(use_async_, 0, num_sortings, [&](unsigned i) {
    auto permute_arr = [&](unsigned i, const auto f) {
      Span<unsigned> indexes(indexes_[i]);
      thrust::for_each(
          thrust_data_[i].execution_policy(),
          thrust::make_counting_iterator(0u),
          thrust::make_counting_iterator(unsigned(indexes_[i].size())),
          [=] __host__ __device__(unsigned k) { f(k, indexes[k]); });
    };

    Span<const IdxAABB> aabbs(aabbs_);

    auto edge_axis = [&](AllEdges &edges,
                         Span<std::array<float, 2>> group_min_maxs,
                         uint8_t axis, uint8_t other_axis) {
      Span<float> other_mins = edges.other_mins();
      Span<float> other_maxs = edges.other_maxs();
      Span<float> values = edges.values();
      Span<uint8_t> is_mins = edges.is_mins();
      unsigned num_edges_per = 2 * num_shapes_;
      Span<float> tree_min = axis == 0 ? min_x_tree_ : min_y_tree_;
      Span<float> tree_max = axis == 0 ? max_x_tree_ : max_y_tree_;

      return [=] __host__ __device__(unsigned k, unsigned index) {
        bool is_min = !bool(index % 2);

        const auto &aabb = aabbs[index / 2].aabb;
        const auto &min_b = aabb.min_bound;
        const auto &max_b = aabb.max_bound;

        other_mins[k] = min_b[other_axis];
        other_maxs[k] = max_b[other_axis];
        float value = is_min ? min_b[axis] : max_b[axis];
        values[k] = value;
        is_mins[k] = is_min;

        if (k % num_edges_per == 0) {
          unsigned group_idx = k / num_edges_per;
          // first in group
          assert(is_min);
          group_min_maxs[group_idx][0] = value;
          unsigned node_idx = group_idx + (axis == 0 ? 0 : num_dir_trees) + 1;
          nodes[node_idx] = DirTreeNode(value, 0, node_idx + num_dir_trees);
          tree_min[group_idx] = value;
        } else if ((k + 1) % num_edges_per == 0) {
          unsigned group_idx = k / num_edges_per;
          // last in group
          assert(!is_min);
          group_min_maxs[group_idx][1] = value;
          unsigned node_idx = group_idx + (axis == 0 ? 0 : num_dir_trees) + 1 +
                              2 * num_dir_trees;
          nodes[node_idx] = DirTreeNode(value, node_idx + num_dir_trees, 0);
          tree_max[group_idx] = value;
        }
      };
    };

    auto z_min_max = [&](bool is_min) {
      ZValues &z_vals = (is_min ? sorted_by_z_min_ : sorted_by_z_max_).first;
      auto x_mins = z_vals.x_mins();
      auto x_maxs = z_vals.x_maxs();
      auto y_mins = z_vals.y_mins();
      auto y_maxs = z_vals.y_maxs();
      auto z_mins = z_vals.z_mins();
      auto z_maxs = z_vals.z_maxs();
      auto idxs = z_vals.idxs();
      Span<float> min_z_tree = min_z_tree_;
      Span<float> max_z_tree = max_z_tree_;
      unsigned num_shapes = num_shapes_;
      return [=] __host__ __device__(unsigned k, unsigned index) {
        const auto &aabb = aabbs[index];
        const auto &mins = aabb.aabb.min_bound;
        const auto &maxs = aabb.aabb.max_bound;
        x_mins[k] = mins.x();
        y_mins[k] = mins.y();
        z_mins[k] = mins.z();
        x_maxs[k] = maxs.x();
        y_maxs[k] = maxs.y();
        z_maxs[k] = maxs.z();
        idxs[k] = aabb.idx;
        if (k % num_shapes == 0) {
          unsigned group_idx = k / num_shapes;
          if (is_min) {
            min_z_tree[group_idx] = mins.z();
          } else {
            max_z_tree[group_idx] = maxs.z();
          }
        }
      };
    };

    if (i == 0) {
      permute_arr(
          0, edge_axis(current_edges_, current_edges_min_max_.get(), 0, 1));
    } else if (i == 1) {
      permute_arr(1, edge_axis(other_edges_, other_edges_min_max_.get(), 1, 0));
    } else if (i == 2) {
      permute_arr(2, z_min_max(true));
    } else {
      permute_arr(3, z_min_max(false));
    }
  });
}
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
