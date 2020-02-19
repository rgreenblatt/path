#include "lib/binary_search.h"
#include "lib/span.h"

#include <gtest/gtest.h>

#include <vector>

TEST(binary_search, binary_search) {
  unsigned size = 64;

  std::vector<float> forward = {
      0,  1,  2,  4,  8,  9,  9,  9,  10, 11, 12, 23, 38, 38, 38, 38,
      38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
      38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
      38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38};

  ASSERT_EQ(forward.size(), size);

  std::vector<float> reverse(size);
  std::reverse_copy(forward.begin(), forward.end(), reverse.begin());

  auto search_test = [&](unsigned start, unsigned end, float target,
                         bool is_increasing, unsigned expected) {
    EXPECT_EQ((binary_search<float>(start, end, target,
                                    is_increasing ? forward : reverse,
                                    is_increasing)),
              expected);
    for (unsigned guess = start; guess < end; guess++) {
      for (unsigned increm = 0; increm < 2 * (end - start); increm++) {
        EXPECT_EQ((binary_search<float, false>(
                      start, end, target, guess, increm,
                      is_increasing ? forward : reverse, is_increasing)),
                  expected);
      }
    }
  };

  search_test(0, 0, -0.2f, true, 0);
  search_test(1, 1, 100.2f, true, 1);
  search_test(8, 8, -100.2f, true, 8);
  search_test(0, 1, -0.2f, true, 0);
  search_test(0, 1, 0.1, true, 1);
  search_test(0, 1, 1.1f, true, 1);
  search_test(0, 1, 10.1f, true, 1);
  search_test(0, 3, -0.2f, true, 0);
  search_test(0, 3, 0.2f, true, 1);
  search_test(0, 3, 1.2f, true, 2);
  search_test(0, 3, 2.2f, true, 3);
  search_test(0, 3, 3.2f, true, 3);
  search_test(0, 3, 9.2f, true, 3);
  search_test(0, 60, -0.2f, true, 0);
  search_test(0, 60, 11.1f, true, 10);
  search_test(0, 60, 13.1f, true, 11);
  search_test(0, 60, 40.0f, true, 60);
  search_test(0, 60, 28.0f, true, 12);

  search_test(0, 0, -0.2f, false, 0);
  search_test(1, 1, 100.2f, false, 1);
  search_test(8, 8, -100.2f, false, 8);
  search_test(0, 1, 28.0f, false, 1);
  search_test(0, 1, 38.1f, false, 0);
  search_test(0, 1, 50.0f, false, 0);
  search_test(0, 64, 50.0f, false, 0);
  search_test(0, 64, 9.5f, false, 56);
  search_test(0, 64, 8.5f, false, 59);
}
