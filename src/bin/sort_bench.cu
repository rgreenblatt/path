#include "ray/detail/accel/aabb.h"
#include <chrono>
#include <dbg.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
  using ray::detail::accel::AABB;
  unsigned size = 1000000;
  thrust::device_vector<AABB> aabbs(size);

  thrust::transform(
      thrust::make_counting_iterator(unsigned(0)),
      thrust::make_counting_iterator(unsigned(size)), aabbs.begin(),
      [] __device__(unsigned v) {
        const unsigned fnv_prime = 16777619u;
        const unsigned fnv_offset_basis = 2166136261u;
        v ^= fnv_offset_basis;
        v *= fnv_prime;
        return AABB(
            Eigen::Vector3f(1, 2, v),
            Eigen::Vector3f(1, 2,
                            v + float(v * 883838) /
                                    std::numeric_limits<unsigned>::max()));
      });

  auto start = std::chrono::high_resolution_clock::now();

  thrust::sort(aabbs.begin(), aabbs.end(),
               [] __device__(const AABB &first, const AABB &second) {
                 return first.get_min_bound().z() > second.get_min_bound().z();
               });

  auto end = std::chrono::high_resolution_clock::now();

  double dur =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();

  dbg(dur);

  return 0;
}
