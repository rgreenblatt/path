#include "intersect/accel/detail/bvh/bvh.h"

#include <iostream>
#include <map>

namespace intersect {
namespace accel {
namespace detail {
namespace bvh {
void check_and_print_stats(SpanSized<const NodeValue> node_values,
                           SpanSized<const AABB> node_aabbs, Settings settings,
                           unsigned objects_vec_size) {
  unsigned min_size = std::numeric_limits<unsigned>::max();
  unsigned max_size = std::numeric_limits<unsigned>::lowest();
  unsigned total_size = 0;
  unsigned total_count = 0;
  std::map<unsigned, unsigned> size_counts;

  for (const NodeValue &node_value : node_values) {
    auto value = node_value.as_rep();
    if (value.type() != NodeType::Items) {
      continue;
    }
    unsigned size = value.get(tag_v<NodeType::Items>).size();
    min_size = std::min(min_size, size);
    max_size = std::max(max_size, size);
    total_size += size;
    ++total_count;
    ++size_counts[size];
  }

  always_assert(settings.target_objects + max_size - 1 <= objects_vec_size);

  if (settings.print_stats) {
    std::cout << "SAH cost: " << sa_heurisitic_cost(node_values, node_aabbs, 0)
              << "\nmin size: " << min_size << "\nmax size: " << max_size
              << "\navg size: " << static_cast<float>(total_size) / total_count
              << "\n";
    for (auto [size, count] : size_counts) {
      std::cout << "size: " << size << " has count " << count << std::endl;
    }
  }
}

float sa_heurisitic_cost_impl(SpanSized<const NodeValue> node_values,
                              SpanSized<const AABB> node_aabbs,
                              float traversal_per_intersect_cost,
                              unsigned start_node) {
  const auto &value = node_values[start_node];
  return value.as_rep().visit_tagged([&](auto tag, const auto &value) {
    if constexpr (tag == NodeType::Items) {
      return value.size();
    } else {
      static_assert(tag == NodeType::Split);
      auto get_cost = [&](unsigned idx) {
        return sa_heurisitic_cost_impl(node_values, node_aabbs,
                                       traversal_per_intersect_cost, idx) *
               node_aabbs[idx].surface_area() /
               node_aabbs[start_node].surface_area();
      };

      return traversal_per_intersect_cost + get_cost(value.left_idx) +
             get_cost(value.right_idx);
    }
  });
}

float sa_heurisitic_cost(SpanSized<const NodeValue> node_values,
                         SpanSized<const AABB> node_aabbs,
                         float traversal_per_intersect_cost) {
  return sa_heurisitic_cost_impl(node_values, node_aabbs,
                                 traversal_per_intersect_cost, 0);
}

} // namespace bvh
} // namespace detail
} // namespace accel
} // namespace intersect
