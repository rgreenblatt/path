#include "lib/cuda/utils.h"

#include <tuple>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
HOST_DEVICE inline std::tuple<unsigned, unsigned, unsigned>
left_right_counts(unsigned index_in_group, unsigned start_inclusive,
                  unsigned open_mins, uint8_t is_min, unsigned total_count) {
  unsigned ends_inclusive = (index_in_group + 1) - start_inclusive;
  unsigned starts_exclusive = start_inclusive - is_min;
  unsigned num_left = open_mins + starts_exclusive;
  unsigned num_right = total_count - ends_inclusive;
  unsigned open_mins_before_right = num_left - ends_inclusive;

  return std::make_tuple(num_left, num_right, open_mins_before_right);
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
