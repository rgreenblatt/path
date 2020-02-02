#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::construct() {
  better_than_no_split_.first->resize(num_groups());
  better_than_no_split_.second->resize(num_groups(), 1);

  x_edges_keys_.resize(current_edges_->size());
  y_edges_keys_.resize(other_edges_->size());
  z_keys_.resize(sorted_by_z_min_.first->size());

  fill_keys();

  scan_edges();

  find_best_edges();

  test_splits();

  filter_others();

  /* auto tranform_start = thrust::make_transform_iterator(sorted_by_x_edges_.begin(), ) */
  /* thrust::inclusive_scan(InputIterator first, InputIterator last,
   * OutputIterator result) */
  /* sorted_by_x_edges_ */
  
  // TODO consider breaking up data to reduce memory access....

  // approach:
  // - compute segmented scan for ends and starts
  //   - some possiblility to optimize around last split...
  //   - some possiblility to optimize segmentation
  // - test all edge choices and reduce to best choice (test both x and y for
  //   now, later test alternating).
  // - using best choice, write out where new division will be and new
  //   start end initial for dimension along which split was done
  // - filter the other edges using a prefix sum etc
  // - filter sorted by z min and z max using a prefix sum etc

  // generalized segmented prefix sum is important...
  // generalized segmented transform is important...
  // look at how thrust does things and see if special casing is important
  // look at using bit fields or uint8_t to store filter condition

  // 1. approach to segmented ___:
  //  - index globally
  //  - operate from there
  //  - fill global index using dynamic kernel launches where needed and
  //   otherwise looping...

  // 2. approach to segmented ___:
  //  - index into thread block
  //  - either entire thread block or start of data per warp
  //    index into warp
  //  - either entire warp or start of data per index
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
