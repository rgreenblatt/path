#include "lib/group.h"
#include "lib/span_convertable_vector.h"

#include <gtest/gtest.h>

#include <string>

TEST(Group, get_previous) {
  {
    std::vector arr = {1u, 2u, 3u, 4u, 5u};

    for (unsigned i = 1; i < arr.size(); i++) {
      EXPECT_EQ(arr[i - 1], get_previous(i, arr));
    }
    EXPECT_EQ(0, get_previous(0, arr));
  }

  {
    std::vector<std::array<unsigned, 2>> arr = {
        {1u, 11u}, {2u, 12u}, {3u, 13u}, {4u, 14u}, {5u, 15u}};
    Span<const std::array<unsigned, 2>> arr_span = arr;

    for (unsigned i = 1; i < arr.size(); i++) {
      EXPECT_EQ(arr[i - 1], get_previous(i, arr_span));
    }
    EXPECT_EQ(0, get_previous(0, arr_span)[0]);
    EXPECT_EQ(0, get_previous(0, arr_span)[1]);
  }

  int a = int();
  EXPECT_EQ(a, 0);
  std::array<unsigned, 2> arr = std::array<unsigned, 2>();
  EXPECT_EQ(arr[0], 0);
  EXPECT_EQ(arr[1], 0);
}
