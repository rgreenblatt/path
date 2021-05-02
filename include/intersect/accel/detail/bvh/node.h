#pragma once

#include "intersect/accel/aabb.h"
#include "lib/bit_utils.h"
#include "lib/start_end.h"
#include "lib/tagged_union.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/sequential_dispatch.h"

namespace intersect {
namespace accel {
namespace detail {
namespace bvh {
struct Split {
  unsigned left_idx;
  unsigned right_idx;

  constexpr bool operator==(const Split &other) const = default;
  constexpr auto operator<=>(const Split &other) const = default;
};

enum class NodeType {
  Split,
  Items,
};

using NodeValueRep = TaggedUnion<NodeType, Split, StartEnd<unsigned>>;

class NodeValue {
public:
  NodeValue() = default;

  constexpr inline explicit NodeValue(NodeValueRep rep) {
    raw_values_ = rep.visit_tagged(
        [&](auto tag, const auto &value) -> std::array<unsigned, 2> {
          if constexpr (tag == NodeType::Split) {
            return {value.left_idx, value.right_idx};
          } else {
            return {value.start, value.end};
          }
        });

    debug_assert((tag_bit_mask & raw_values_[0]) == 0);
    if (rep.type() == NodeType::Split) {
      raw_values_[0] |= tag_bit_mask;
    }
  }

  constexpr inline NodeValueRep as_rep() const {
    return sequential_dispatch<2>(
        (raw_values_[0] & tag_bit_mask) >> tag_bit_idx,
        [&]<unsigned idx>(NTag<idx>) -> NodeValueRep {
          if constexpr (idx == 0) {
            return {tag_v<NodeType::Items>,
                    {.start = raw_values_[0], .end = raw_values_[1]}};
          } else {
            static_assert(idx == 1);
            return {tag_v<NodeType::Split>,
                    {.left_idx = raw_values_[0] & non_tag_bit_mask,
                     .right_idx = raw_values_[1]}};
          }
        });
  }

private:
  static constexpr unsigned tag_bit_idx = 31;
  static constexpr unsigned tag_bit_mask = bit_mask<unsigned>(tag_bit_idx);
  static constexpr unsigned non_tag_bit_mask =
      up_to_mask<unsigned>(tag_bit_idx - 1);

  std::array<unsigned, 2> raw_values_;
};
} // namespace bvh
} // namespace detail
} // namespace accel
} // namespace intersect
