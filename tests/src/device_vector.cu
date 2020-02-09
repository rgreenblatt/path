#include "lib/vector_type.h"

#include <gtest/gtest.h>

TEST(DeviceVector, no_default_init) {
  class OneIfDefaultConstructed {
  public:
    unsigned val;

    HOST_DEVICE OneIfDefaultConstructed() { val = 1; }
  };

  unsigned size = 10;

  {
    DeviceVector<OneIfDefaultConstructed> vals(size);

    // technically this could fail...
    for (const auto &v : vals) {
      ASSERT_NE(OneIfDefaultConstructed(v).val, 1u);
    }
  }

  {
    thrust::device_vector<OneIfDefaultConstructed> vals(size);

    for (const auto &v : vals) {
      ASSERT_EQ(OneIfDefaultConstructed(v).val, 1u);
    }
  }
}

TEST(DeviceVector, expand_preserves) {
  unsigned first_size = 1000;
  unsigned next_size = 100000;

  DeviceVector<unsigned> vals(first_size, 0);
  vals.resize(next_size, 1);

  ASSERT_EQ(vals[0], 0u);
  ASSERT_EQ(vals[first_size - 1], 0u);
  ASSERT_EQ(vals[first_size], 1u);
  ASSERT_EQ(vals[next_size - 1], 1u);
}
