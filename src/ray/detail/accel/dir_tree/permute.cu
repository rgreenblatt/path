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
  async_for<true>(0, num_sortings, [&](unsigned i) {
    auto permute_arr = [&](unsigned i, const auto f) {
      Span<unsigned> indexes(indexes_[i]);
      thrust::for_each(
          thrust_data_[i].execution_policy(),
          thrust::make_counting_iterator(0u),
          thrust::make_counting_iterator(unsigned(indexes_[i].size())),
          [=] __host__ __device__(unsigned k) { f(k, indexes[k]); });
    };

    Span<const IdxAABB> aabbs(aabbs_);

    auto edge_axis = [&](AllEdges &edges, uint8_t axis, uint8_t other_axis) {
      Span<float> other_mins = edges.other_mins();
      Span<float> other_maxs = edges.other_maxs();
      Span<float> values = edges.values();
      Span<uint8_t> is_mins = edges.is_mins();

      return [=] __host__ __device__(unsigned k, unsigned index) {
        bool is_min = !bool(index % 2);

        const auto &aabb = aabbs[index / 2].aabb;
        const auto &min_b = aabb.get_min_bound();
        const auto &max_b = aabb.get_max_bound();

        other_mins[k] = min_b[other_axis];
        other_maxs[k] = max_b[other_axis];
        values[k] = is_min ? min_b[axis] : max_b[axis];
        is_mins[k] = is_min;
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
      return [=] __host__ __device__(unsigned k, unsigned index) {
        const auto &aabb = aabbs[index];
        const auto &mins = aabb.aabb.get_min_bound();
        const auto &maxs = aabb.aabb.get_max_bound();
        x_mins[k] = mins.x();
        y_mins[k] = mins.y();
        z_mins[k] = mins.z();
        x_maxs[k] = maxs.x();
        y_maxs[k] = maxs.y();
        z_maxs[k] = maxs.z();
        idxs[k] = aabb.idx;
      };
    };

    if (i == 0) {
      permute_arr(0, edge_axis(x_edges_.first, 0, 1));
    } else if (i == 1) {
      permute_arr(1, edge_axis(y_edges_.first, 1, 0));
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
