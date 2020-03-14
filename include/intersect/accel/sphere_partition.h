#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <Eigen/Core>

#include <assert.h>
#include <tuple>
#include <vector>

namespace intersect {
namespace accel {
// TODO test!!!
class HalfSpherePartition {
public:
  // colatitude is from 0 - pi
  // 0 when vector is straight forward (x = 0, y = 0, z = 1)
  // pi when vector is straight back

  // longitude is from 0 - 2 pi
  // 0 when vector is straight in x direction (x = 1, y = 0, z = 0)
  // +/- pi when vector is away in x direction (x = -1, y = 0, z = 0)
  // pi / 2 when vector is toward y (x = 0, y = 1, z = 0)
  // -pi / 2 when vector is away in y (x = 0, y = -1, z = 0)
  HOST_DEVICE static std::tuple<float, float>
  vec_to_colatitude_longitude(const Eigen::Vector3f &vec);

  // inverse of above
  HOST_DEVICE static Eigen::Vector3f
  colatitude_longitude_to_vec(float colatitude, float longitude);

  HOST_DEVICE std::tuple<float, float>
  get_center_colatitude_longitude(unsigned collar, unsigned region) const;

  HOST_DEVICE Eigen::Vector3f get_center_vec(unsigned collar,
                                             unsigned region) const;

  // colatitude should be 0 - pi
  // longitude should be 0 - 2 * pi
  HOST_DEVICE std::tuple<unsigned, bool> get_closest(float colatitude,
                                                     float longitude) const;

  HOST_DEVICE std::tuple<unsigned, bool>
  get_closest(const Eigen::Vector3f &vec) const;

  HOST_DEVICE inline unsigned size() const {
    return colatitude_divs_[colatitude_divs_.size() - 1].end_index;
  }

  struct ColatitudeDiv {
    float inverse_interval;
    unsigned start_index;
    unsigned end_index;

    ColatitudeDiv(float inverse_interval, unsigned start_index,
                  unsigned end_index)
        : inverse_interval(inverse_interval), start_index(start_index),
          end_index(end_index) {}

    ColatitudeDiv() {}
  };

  inline SpanSized<const ColatitudeDiv> colatitude_divs() const {
    return colatitude_divs_;
  }

  template <typename T>
  HalfSpherePartition(unsigned target_num_regions, T &regions);

  HalfSpherePartition() {}

private:
  float colatitude_offset_;
  float colatitude_inverse_interval_;
  SpanSized<const ColatitudeDiv> colatitude_divs_;
};
} // namespace accel
} // namespace intersect
