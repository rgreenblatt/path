#include "lib/cuda/utils.h"
#include "lib/printf_dbg.h"
#include "lib/span.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/thrust_data.h"

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "dbg.h"

TEST(Scan, check_calls) {
  for (unsigned size : {10, 101, 1000, 10383, 1023838}) {
    ThrustData<ExecutionModel::GPU> gpu_thrust_data;
    ThrustData<ExecutionModel::CPU> cpu_thrust_data;

    thrust::device_vector<unsigned> device_incrementer(1);
    unsigned host_incrementer = 0;

    thrust::fill(device_incrementer.begin(), device_incrementer.end(), 0);

    unsigned *host_inc = &host_incrementer;
    Span<unsigned> device_inc = device_incrementer;

    thrust::device_vector<unsigned> device_out(size);
    std::vector<unsigned> host_out(size);

    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0u),
        [=] __host__ __device__(unsigned i) -> unsigned {
#ifdef __CUDA_ARCH__
          device_inc[0]++;
#else
          (*host_inc)++;
#endif
          return 1u;
        });

    thrust::inclusive_scan(gpu_thrust_data.execution_policy(), it, it + size,
                           device_out.begin());
    thrust::inclusive_scan(cpu_thrust_data.execution_policy(), it, it + size,
                           host_out.data());

    ASSERT_EQ(device_incrementer[0], size);
    ASSERT_EQ(host_incrementer, size);
  }
}
