#include "lib/edges.h"
#include "lib/span.h"

#include <gtest/gtest.h>

#include <vector>

TEST(edges, edges_get_previous) {
  std::vector arr = {1u, 2u, 3u, 4u, 5u};

  for (unsigned i = 1; i < arr.size(); i++) {
    EXPECT_EQ(arr[i - 1], edges_get_previous<unsigned>(i, arr));
  }
  EXPECT_EQ(0, edges_get_previous<unsigned>(0, arr));
}
