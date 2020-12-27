#include "intersect/accel/kdtree/generator.h"
#include "lib/assert.h"
#include "lib/eigen_utils.h"

namespace intersect {
namespace accel {
namespace kdtree {
using namespace detail;

// inspired by https://www.geeksforgeeks.org/quickselect-algorithm/
template <ExecutionModel execution_model>
unsigned KDTree<execution_model>::Generator::partition(unsigned start,
                                                       unsigned end,
                                                       uint8_t axis) {
  float x = bounds_[end].center[axis];
  size_t i = start;
  for (size_t j = start; j <= end - 1; j++) {
    if (bounds_[j].center[axis] <= x) {
      std::swap(bounds_[i], bounds_[j]);
      std::swap(indexes_[i], indexes_[j]);
      i++;
    }
  }
  std::swap(bounds_[i], bounds_[end]);
  std::swap(indexes_[i], indexes_[end]);

  return i;
}

// inspired by https://www.geeksforgeeks.org/quickselect-algorithm/
template <ExecutionModel execution_model>
void KDTree<execution_model>::Generator::kth_smallest(size_t start, size_t end,
                                                      size_t k, uint8_t axis) {
  // Partition the array around last
  // element and get position of pivot
  // element in sorted array
  size_t index = partition(start, end, axis);

  // If position is same as k
  if (index - start == k || (index - start == 1 && k == 0) ||
      (end - index == 1 && k == end - start)) {
    return;
  }

  // If position is more, recur
  // for left subarray
  if (index - start > k) {
    return kth_smallest(start, index - 1, k, axis);
  }

  // Else recur for right subarray
  kth_smallest(index + 1, end, k - index + start - 1, axis);
}

template <ExecutionModel execution_model>
bool KDTree<execution_model>::Generator::terminate_here(unsigned start,
                                                        unsigned end) {
  return end - start <= settings_.num_objects_terminate;
  /* || settings_.use_s_a_heuritic */ // not yet supported
}

template <ExecutionModel execution_model>
AABB KDTree<execution_model>::Generator::get_bounding(unsigned start,
                                                      unsigned end) {
  debug_assert(start != end);

  AABB out = bounds_[start].aabb;
  for (unsigned i = start + 1; i < end; i++) {
    out = out.union_other(bounds_[i].aabb);
  }

  return out;
}

template <ExecutionModel execution_model>
unsigned KDTree<execution_model>::Generator::construct(unsigned start,
                                                       unsigned end,
                                                       unsigned depth) {
  debug_assert(start != end);
  if (terminate_here(start, end)) {
    auto total_bounds = get_bounding(start, end);
    unsigned index = nodes_.size();
    nodes_.push_back(KDTreeNode({start, end}, total_bounds));

    return index;
  }

  const unsigned axis = depth % 3;
  const size_t k = (end - start) / 2;
  kth_smallest(start, end - 1, k, axis);
  float median = bounds_[k + start].center[axis];
  unsigned new_depth = depth + 1;
  unsigned left_index, right_index;
  left_index = construct(start, start + k, new_depth);
  right_index = start + k == end ? 0 : construct(start + k, end, new_depth);
  auto &left = nodes_[left_index];
  auto &right = nodes_[right_index];

  unsigned index = nodes_.size();
  nodes_.push_back(
      KDTreeNode(KDTreeSplit(left_index, right_index, median),
                 left.get_contents().union_other(right.get_contents())));

  return index;
}

template <ExecutionModel execution_model>
std::tuple<SpanSized<const KDTreeNode<AABB>>, Span<const unsigned>>
KDTree<execution_model>::Generator::gen(const Settings &settings,
                                        SpanSized<Bounds> bounds) {
  settings_ = settings;
  settings_.num_objects_terminate =
      std::max(settings_.num_objects_terminate, 1u);

  nodes_.clear();
  nodes_out_.clear();
  indexes_out_.clear();

  if (bounds.size() == 0) {
    return {nodes_out_, indexes_out_};
  }

  indexes_.resize(bounds.size());
  bounds_ = bounds;

  for (unsigned i = 0; i < bounds.size(); ++i) {
    indexes_[i] = i;
  }

  nodes_.push_back(KDTreeNode({0, 0}, AABB{max_eigen_vec(), min_eigen_vec()}));

  construct(0, bounds.size(), 0);

  if constexpr (execution_model == ExecutionModel::GPU) {
    nodes_out_.resize(nodes_.size());
    indexes_out_.resize(indexes_.size());
    thrust::copy(nodes_.data(), nodes_.data() + nodes_.size(),
                 nodes_out_.begin());
    thrust::copy(indexes_.data(), indexes_.data() + indexes_.size(),
                 indexes_out_.begin());

    return {nodes_out_, indexes_out_};
  } else {
    return {nodes_, indexes_};
  }
}

template class KDTree<ExecutionModel::CPU>::Generator;
template class KDTree<ExecutionModel::GPU>::Generator;
} // namespace kdtree
} // namespace accel
} // namespace intersect
