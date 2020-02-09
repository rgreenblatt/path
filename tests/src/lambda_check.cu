#include "lib/span.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/thrust_data.h"
#include "lib/vector_type.h"

#include <gtest/gtest.h>

// running into some weird issues with lambda capture
// I think they only show up when conditionally capturing variables
// using #ifdef __CUDA_ARCH__
TEST(LambdaCaptureWorks, list_args) {
  unsigned size = 1000;
  auto start_it = thrust::make_counting_iterator(0u);
  ThrustData<ExecutionModel::GPU> thrust_data;

  {
    DeviceVector<unsigned> vals_first(size);
    DeviceVector<float> vals_second(size);

    thrust::for_each(
        thrust_data.execution_policy(), start_it, start_it + size,
        [f = Span<unsigned>(vals_first),
         s = Span<float>(vals_second)] __device__(unsigned i) {
          f[i] = i;
          s[i] = i;
        });

    for (unsigned i = 0; i < size; ++i) {
      ASSERT_EQ(vals_first[i], i);
      ASSERT_EQ(vals_second[i], float(i));
    }
  }

  {
    DeviceVector<unsigned> vals_first(size);
    DeviceVector<float> vals_second(size);

    Span<unsigned> f = vals_first;
    Span<float> s = vals_second;

    thrust::for_each(thrust_data.execution_policy(), start_it, start_it + size,
                     [f, s] __device__(unsigned i) {
                       f[i] = i;
                       s[i] = i;
                     });

    for (unsigned i = 0; i < size; ++i) {
      ASSERT_EQ(vals_first[i], i);
      ASSERT_EQ(vals_second[i], float(i));
    }
  }

  {
    DeviceVector<unsigned> vals_first(size);
    DeviceVector<float> vals_second(size);

    Span<unsigned> f = vals_first;
    Span<float> s = vals_second;

    thrust::for_each(start_it, start_it + size,
                     [=] __device__(unsigned i) {
                       f[i] = i;
                       s[i] = i;
                     });

    for (unsigned i = 0; i < size; ++i) {
      ASSERT_EQ(vals_first[i], i);
      ASSERT_EQ(vals_second[i], float(i));
    }
  }

  {
    DeviceVector<unsigned> vals_first(size);
    DeviceVector<float> vals_second(size);
    DeviceVector<unsigned> out(size);

    Span<unsigned> f = vals_first;
    Span<float> s = vals_second;

    auto start_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0u),
        [=] __host__ __device__(unsigned i) -> unsigned {
          f[i] = i;
          s[i] = i;

          return i;
        });

    thrust::inclusive_scan(thrust_data.execution_policy(), start_it,
                           start_it + size, out.begin());

    for (unsigned i = 0; i < size; ++i) {
      ASSERT_EQ(vals_first[i], i);
      ASSERT_EQ(vals_second[i], float(i));
    }
  }
}
