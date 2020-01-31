#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

#include <thrust/iterator/permutation_iterator.h>

#include <future>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::permute() {
  std::array<std::future<void>, num_sortings> results;

  auto permute_arr = [&](unsigned i, const auto f) {
    results[i] = std::async(std::launch::async, [&]() {
      Span<unsigned> indexes(indexes_[i]);
      thrust::for_each(
          thrust_data_[i].execution_policy(),
          thrust::make_counting_iterator(0u),
          thrust::make_counting_iterator(unsigned(indexes_[i].size())),
          [=] __host__ __device__(unsigned k) { f(k, indexes[k]); });
    });
  };

  Span<const IdxAABB> aabbs(aabbs_);

  auto edge_axis = [&](Span<float> other_mins, Span<float> other_maxs,
                       Span<float> values, 
                       Span<uint8_t> is_mins, uint8_t axis, uint8_t other_axis) {
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

  permute_arr(0,
              edge_axis(x_edges_.template get<0>(), x_edges_.template get<1>(),
                        x_edges_.template get<2>(), x_edges_.template get<3>(),
                        0, 1));
  permute_arr(1,
              edge_axis(y_edges_.template get<0>(), y_edges_.template get<1>(),
                        y_edges_.template get<2>(), y_edges_.template get<3>(),
                        1, 0));

  auto z_min_max = [&](bool is_min) {
    Span<IdxAABB> sorted_by_z(is_min ? sorted_by_z_min_ : sorted_by_z_max_);
    return [=] __host__ __device__(unsigned k, unsigned index) {
      sorted_by_z[k] = aabbs[index];
    };
  };

  permute_arr(2, z_min_max(true));
  permute_arr(3, z_min_max(false));

  for (auto &result : results) {
    result.get();
  }
}
template class DirTreeGenerator<ExecutionModel::GPU>;
template class DirTreeGenerator<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
