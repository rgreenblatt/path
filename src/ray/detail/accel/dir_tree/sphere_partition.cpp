#include "ray/detail/accel/dir_tree/sphere_partition.h"

#include <cmath>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
inline float area_of_cap(float s_cap) {
  const float sin_v = std::sin(s_cap / 2);

  return 4 * float(M_PI) * sin_v * sin_v;
}

HalfSpherePartition::HalfSpherePartition(unsigned target_num_regions,
                                         ManangedMemVec<Region> &regions) {
  const float area_of_half_sphere = 2 * float(M_PI);
  const float area_of_ideal_region = area_of_half_sphere / target_num_regions;

  const float polar_colatitude =
      2 * std::asin(std::sqrt(area_of_ideal_region / float(M_PI)) / 2);
  const float ideal_collar_angle = std::pow(area_of_ideal_region, 0.5f);

  const float angle_space = float(M_PI) / 2 - polar_colatitude;

  // check this comp (max etc...)
  const unsigned n_collars = std::round(angle_space / ideal_collar_angle);

  const float fitting_angle = angle_space / n_collars;

  std::vector<unsigned> num_regions(n_collars);

  for (unsigned collar_num = 0; collar_num < n_collars; collar_num++) {
    // difference between above and below is area of collar
    float ideal_collar_area =
        area_of_cap(polar_colatitude + (collar_num + 1) * fitting_angle) -
        area_of_cap(polar_colatitude + collar_num * fitting_angle);
    num_regions[collar_num] =
        std::round(ideal_collar_area / area_of_ideal_region);
  }

  colatitude_offset_ = fitting_angle - polar_colatitude;

  colatitude_inverse_interval_ = 1.0f / fitting_angle;

  regions.resize(n_collars + 1);

  unsigned total_num_regions = 0;
  regions[0] = Region(0.0, 0, 1);
  total_num_regions++;
  for (unsigned collar_num = 0; collar_num < n_collars; collar_num++) {
    unsigned this_num_regions = num_regions[collar_num];
    regions[collar_num] =
        Region(this_num_regions / 2.0f * float(M_PI), total_num_regions,
               total_num_regions + this_num_regions);
    total_num_regions += this_num_regions;
  }

  regions_ = Span<const Region, false>(regions.data(), regions.size());
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
