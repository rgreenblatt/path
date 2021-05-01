#include "intersect/accel/detail/bvh/node.h"

#include <gtest/gtest.h>

using namespace intersect::accel::detail::bvh;

TEST(NodeValue, convert) {
  auto check_identity = [](NodeValueRep rep) {
    EXPECT_EQ(NodeValue(rep).as_rep(), rep);
  };

  check_identity(NodeValueRep{tag_v<NodeType::Split>,
                              {.left_idx = 2838382, .right_idx = 82}});
  check_identity(
      NodeValueRep{tag_v<NodeType::Split>,
                   {.left_idx = 2147483647, .right_idx = 2147483647}});
  check_identity(NodeValueRep{tag_v<NodeType::Split>,
                              {.left_idx = 0, .right_idx = 2147483647}});
  check_identity(NodeValueRep{tag_v<NodeType::Split>,
                              {.left_idx = 2147483647, .right_idx = 0}});
  check_identity(
      NodeValueRep{tag_v<NodeType::Split>, {.left_idx = 0, .right_idx = 0}});

  check_identity(
      NodeValueRep{tag_v<NodeType::Items>, {.start = 2838382, .end = 82}});
  check_identity(NodeValueRep{tag_v<NodeType::Items>,
                              {.start = 2147483647, .end = 2147483647}});
  check_identity(
      NodeValueRep{tag_v<NodeType::Items>, {.start = 0, .end = 2147483647}});
  check_identity(
      NodeValueRep{tag_v<NodeType::Items>, {.start = 2147483647, .end = 0}});
  check_identity(NodeValueRep{tag_v<NodeType::Items>, {.start = 0, .end = 0}});
}
