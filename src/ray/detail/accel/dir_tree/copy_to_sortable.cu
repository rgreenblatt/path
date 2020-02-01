#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/block_data.h"
#include "ray/detail/render_impl_utils.h"

#include <future>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::copy_to_sortable() {
  Span<IdxAABB> aabbs(aabbs_);

  Span<Eigen::Vector3f> offsets(sort_offsets_);
  Span<float> x_edges(sorting_values_[0]);
  Span<float> y_edges(sorting_values_[1]);
  Span<float> z_mins(sorting_values_[2]);
  Span<float> z_maxs(sorting_values_[3]);

  thrust::for_each(
      thrust_data_[0].execution_policy(), thrust::make_counting_iterator(0u),
      thrust::make_counting_iterator(unsigned(aabbs_.size())),
      [=, num_shapes = num_shapes_] __host__ __device__(unsigned idx) {
        Eigen::Vector3f offset = offsets[idx / num_shapes];

        Eigen::Vector3f min_v = aabbs[idx].aabb.get_min_bound() + offset;
        Eigen::Vector3f max_v = aabbs[idx].aabb.get_max_bound() + offset;

        unsigned min_edge_index = 2 * idx;
        unsigned max_edge_index = 2 * idx + 1;

        x_edges[min_edge_index] = min_v.x();
        x_edges[max_edge_index] = max_v.x();
        y_edges[min_edge_index] = min_v.y();
        y_edges[max_edge_index] = max_v.y();
        z_mins[idx] = min_v.z();
        z_maxs[idx] = max_v.z();
      });
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
