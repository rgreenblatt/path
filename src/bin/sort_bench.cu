#include "lib/cuda/utils.h"
#include "lib/caching_thrust_allocator.h"
#include "ray/detail/accel/aabb.h"
#include "ray/detail/render_impl_utils.h"


#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <future>

#include <chrono>
#include <dbg.h>

int main() {
  using ray::detail::accel::AABB;
  unsigned size_per = 1000000;
  unsigned blocks = 200;
  /* unsigned size_per = 10000000; */
  /* unsigned blocks = 1; */
  unsigned size = size_per * blocks;
  thrust::device_vector<uint16_t> values(size);
  thrust::device_vector<unsigned> indexes(size);
  thrust::device_vector<AABB> aabbs(size);
  thrust::device_vector<AABB> final_aabbs(size);

  std::vector<
      std::pair<CachingThrustAllocator<ExecutionModel::GPU>, cudaStream_t>>
      thrust_data(blocks);

  for (auto &v : thrust_data) {
    cudaStreamCreate(&v.second);
  }

  auto fill = [&] {
    thrust::transform(thrust::make_counting_iterator(unsigned(0)),
                      thrust::make_counting_iterator(unsigned(size)),
                      values.begin(), [] __device__(unsigned v) {
                        const unsigned fnv_prime = 104729;
                        const unsigned fnv_offset_basis = 2166136261u;
                        v ^= fnv_offset_basis;
                        v *= fnv_prime;

                        return v;
                      });
  };

  auto sort = [&](bool show_times) {
    thrust::copy(thrust::device, thrust::make_counting_iterator(0u),
                 thrust::make_counting_iterator(size), indexes.begin());

    auto sort_block = [&](unsigned block) {
      return [&, block] {
        unsigned start = block * size_per;
        unsigned end = (block + 1) * size_per;
        auto &[alloc, c_stream] = thrust_data[block];
        thrust::sort_by_key(
            /* thrust::device, */
            thrust::cuda::par(alloc).on(c_stream), values.begin() + start,
            values.begin() + end, indexes.begin() + start);
      };
    };

    auto start_sort = std::chrono::high_resolution_clock::now();
#if 1
    std::vector<std::future<void>> results(blocks);

    for (unsigned block = 0; block < blocks; block++) {
      results[block] = std::async(std::launch::async, sort_block(block));
    }

    for (unsigned block = 0; block < blocks; block++) {
      results[block].get();
    }
#else
    for (unsigned block = 0; block < blocks; block++) {
      sort_block(block)();
    }
#endif
    auto end_sort = std::chrono::high_resolution_clock::now();

    double dur_sort = std::chrono::duration_cast<std::chrono::duration<double>>(
                          end_sort - start_sort)
                          .count();

    if (show_times) {
      dbg(dur_sort);
      dbg(size / dur_sort);
    }

    auto start_permute = std::chrono::high_resolution_clock::now();

    thrust::copy(
        thrust::device,
        thrust::make_permutation_iterator(aabbs.begin(), indexes.begin()),
        thrust::make_permutation_iterator(aabbs.end(), indexes.end()),
        final_aabbs.begin());

    auto end_permute = std::chrono::high_resolution_clock::now();

    double dur_permute =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_permute -
                                                                  start_permute)
            .count();

    if (show_times) {
      dbg(dur_permute);
    }
  };

  fill();
  sort(false);
  fill();
  sort(false);
  fill();
  sort(false);
  fill();
  sort(false);
  fill();

  sort(true);

  sort(true);

  return 0;
}
