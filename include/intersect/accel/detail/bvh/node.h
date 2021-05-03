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

class PackedBoolUnsigned {
public:
  PackedBoolUnsigned() = default;

  constexpr inline PackedBoolUnsigned(bool bool_value,
                                      unsigned unsigned_value) {
    raw_value_ = unsigned_value;
    debug_assert((tag_bit_mask & raw_value_) == 0);
    if (bool_value) {
      raw_value_ |= tag_bit_mask;
    }
  }

  constexpr inline unsigned unsigned_value() const {
    return raw_value_ & non_tag_bit_mask;
  }

  constexpr inline bool bool_value() const {
    return (raw_value_ & tag_bit_mask) != 0;
  }

private:
  static constexpr unsigned tag_bit_idx = 31;
  static constexpr unsigned tag_bit_mask = bit_mask<unsigned>(tag_bit_idx);
  static constexpr unsigned non_tag_bit_mask =
      up_to_mask<unsigned>(tag_bit_idx - 1);

  unsigned raw_value_;
};

enum class NodeType {
  Split,
  Items,
};

struct Items {
#ifndef NDEBUG
  // work around for https://bugs.llvm.org/show_bug.cgi?id=50203
  size_t pad = 0;
#endif
  bool is_for_extra;
  StartEnd<unsigned> start_end;

  constexpr bool operator==(const Items &other) const = default;
  constexpr auto operator<=>(const Items &other) const = default;
};

using NodeValueRep = TaggedUnion<NodeType, Split, Items>;

class NodeValue {
public:
  NodeValue() = default;

  constexpr inline explicit NodeValue(NodeValueRep rep) {
    auto values = rep.visit_tagged(
        [&](auto tag, const auto &value) -> std::array<unsigned, 2> {
          if constexpr (tag == NodeType::Split) {
            return {value.left_idx, value.right_idx};
          } else {
            return {value.start_end.start, value.start_end.end};
          }
        });

    first_value_ = PackedBoolUnsigned(rep.type() == NodeType::Split, values[0]);
    second_value_ =
        PackedBoolUnsigned(rep.type() == NodeType::Items &&
                               rep.get(tag_v<NodeType::Items>).is_for_extra,
                           values[1]);
  }

  constexpr inline NodeValueRep as_rep() const {
    return sequential_dispatch<2>(
        first_value_.bool_value(),
        [&]<unsigned idx>(NTag<idx>) -> NodeValueRep {
          if constexpr (idx == 0) {
            return {
                tag_v<NodeType::Items>,
                {
                    .is_for_extra = second_value_.bool_value(),
                    .start_end =
                        {
                            .start = first_value_.unsigned_value(),
                            .end = second_value_.unsigned_value(),
                        },
                },
            };
          } else {
            static_assert(idx == 1);
            debug_assert(!second_value_.bool_value());
            return {tag_v<NodeType::Split>,
                    {
                        .left_idx = first_value_.unsigned_value(),
                        .right_idx = second_value_.unsigned_value(),
                    }};
          }
        });
  }

private:
  PackedBoolUnsigned first_value_;
  PackedBoolUnsigned second_value_;
};

struct Node {
  NodeValue value;
  AABB aabb;
};
} // namespace bvh
} // namespace detail
} // namespace accel
} // namespace intersect
