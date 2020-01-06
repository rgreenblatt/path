#pragma once

#include "lib/cuda/unified_memory_vector.h"
#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <Eigen/Core>

#include <assert.h>
#include <tuple>
#include <vector>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
// TODO test!!!
class HalfSpherePartition {
public:
  static std::tuple<float, float>
  vec_to_colatitude_longitude(const Eigen::Vector3f &vec);

  // inverse of above
  static Eigen::Vector3f colatitude_longitude_to_vec(float colatitude,
                                                     float longitude);

  std::tuple<float, float>
  get_center_colatitude_longitude(unsigned collar, unsigned region) const;

  Eigen::Vector3f get_center_vec(unsigned collar, unsigned region) const;

  std::tuple<unsigned, unsigned> index_to_collar_region(unsigned index) const;

  // colatitude should be 0 - pi
  // longitude should be 0 - 2 * pi
  HOST_DEVICE unsigned get_closest(float colatitude, float longitude) const;

  // colatitude should be 0 - pi
  // longitude should be 0 - 2 * pi
  HOST_DEVICE unsigned get_closest(const Eigen::Vector3f &vec) const;

  HOST_DEVICE inline unsigned size() const {
    return regions_[regions_.size() - 1].end_index;
  }

  struct Region {
    float inverse_interval;
    unsigned start_index;
    unsigned end_index;

    Region(float inverse_interval, unsigned start_index, unsigned end_index)
        : inverse_interval(inverse_interval), start_index(start_index),
          end_index(end_index) {}

    Region() {}
  };

  inline Span<const Region, false> regions() const { return regions_; }

  HalfSpherePartition(unsigned target_num_regions,
                      ManangedMemVec<Region> &regions);

  HalfSpherePartition() {}

private:
  float colatitude_offset_;
  float colatitude_inverse_interval_;
  Span<const Region, false> regions_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
