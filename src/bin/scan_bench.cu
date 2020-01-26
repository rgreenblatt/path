#include "lib/bitset.h"
#include "lib/caching_thrust_allocator.h"
#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/timer.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include <array>
#include <dbg.h>
#include <future>
#include <numeric>
#include <thrust/scan.h>

int main() {
  std::array sizes = {10000000};

  unsigned total_size = std::accumulate(sizes.begin(), sizes.end(), 0);

  using ray::detail::accel::dir_tree::Edge;

  std::vector<Edge> cpu_edges(total_size);

  std::generate(cpu_edges.begin(), cpu_edges.end(),
                [] { return Edge(0, 0, 0, rand() % 2); });

  std::vector<ThrustData<ExecutionModel::GPU>> thrust_data(sizes.size());

  thrust::device_vector<Edge> edges(cpu_edges.begin(), cpu_edges.end());

  thrust::device_vector<unsigned> keys(total_size);

  {
    unsigned total = 0;
    for (unsigned i = 0; i < sizes.size(); i++) {
      unsigned size = sizes[i];
      thrust::fill(keys.begin() + total, keys.begin() + size + total, i);
      total += size;
    }
  }

  thrust::device_vector<unsigned> bit_keys((total_size + 32 - 1) / 32);
  BitSetRef<unsigned> bit_set_keys(bit_keys, total_size);

  {
    unsigned total = 0;
    for (unsigned i = 0; i < sizes.size(); i++) {
      unsigned size = sizes[i];
      unsigned start_location_inside =
          (total + bit_set_keys.bits_per_block - 1) /
          bit_set_keys.bits_per_block;
      unsigned mod_v = total % bit_set_keys.bits_per_block;
      bool bit = i % 2;
      if (mod_v != 0) {
        unsigned unfinished_idx = start_location_inside - 1;
        unsigned mask_up_to = bit_set_keys.up_to_mask(mod_v);
        bit_keys[unfinished_idx] =
            bit_keys[unfinished_idx] & mask_up_to + bit ? ~mask_up_to : 0u;
      }
      unsigned end_location_inside =
          (total + size + bit_set_keys.bits_per_block - 1) /
          bit_set_keys.bits_per_block;
      if (end_location_inside > start_location_inside) {
        thrust::fill(bit_keys.begin() + start_location_inside,
                     bit_keys.begin() + end_location_inside,
                     bit ? std::numeric_limits<float>::max() : 0u);
      }
      total += size;
    }
  }

  thrust::device_vector<uint32_t> counts(total_size);

  auto convert_edge_to_sumable =
      [] __host__ __device__(const Edge &edge) -> uint32_t {
    return edge.is_min;
  };

  auto start_transform_iter =
      thrust::make_transform_iterator(edges.begin(), convert_edge_to_sumable);
  auto end_transform_iter = start_transform_iter + total_size;

  auto transform_full_scan = [&] {
    thrust::inclusive_scan(thrust_data[0].execution_policy(),
                           start_transform_iter, end_transform_iter,
                           counts.begin());
  };

  transform_full_scan();
  transform_full_scan();
  transform_full_scan();
  transform_full_scan();

  Timer transform_full_scan_timer;
  transform_full_scan();
  transform_full_scan_timer.report("transform full scan");

  auto transform_by_keys = [&] {
    thrust::inclusive_scan_by_key(thrust_data[0].execution_policy(),
                                  keys.begin(), keys.end(),
                                  start_transform_iter, counts.begin());
  };

  transform_by_keys();
  transform_by_keys();
  transform_by_keys();
  transform_by_keys();

  Timer transform_by_keys_timer;
  transform_by_keys();
  transform_by_keys_timer.report("transform by keys");

  thrust::device_vector<uint32_t> vals_32(start_transform_iter,
                                          end_transform_iter);
  thrust::device_vector<uint8_t> vals_8(start_transform_iter,
                                        end_transform_iter);

  auto full_scan_bench = [&](const auto &vals, const std::string &s) {
    auto full_scan = [&] {
      thrust::inclusive_scan(thrust_data[0].execution_policy(), vals.begin(),
                             vals.end(), counts.begin());
    };

    full_scan();
    full_scan();
    full_scan();
    full_scan();

    Timer full_scan_timer;
    full_scan();
    full_scan_timer.report(s);
  };

  full_scan_bench(vals_32, "full scan 32");
  full_scan_bench(vals_8, "full scan 8");

  return 0;
}
