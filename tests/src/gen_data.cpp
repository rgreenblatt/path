#include "generate_data/gen_data.h"

#include <gtest/gtest.h>

using namespace generate_data;

TEST(gen_data, gen_data) {
  gen_data(1, 1, 1);
  gen_data(2, 1, 1);
  gen_data(3, 1, 1);
  gen_data(4, 1, 1);
}
