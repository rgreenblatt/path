#include "ray/detail/accel/dir_tree/sphere_partition.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
HOST_DEVICE inline std::tuple<float, float>
HalfSpherePartition::vec_to_colatitude_longitude(const Eigen::Vector3f &vec) {
  // maybe input should always be normalized (so normalizing here isn't
  // required)
  const auto normalized = vec.normalized().eval();
  float colatitude = M_PI / 2 - std::asin(normalized.z());
  float longitude = std::atan2(normalized.y(), normalized.x());

  return std::make_tuple(colatitude, longitude);
}

// inverse of above
HOST_DEVICE inline Eigen::Vector3f
HalfSpherePartition::colatitude_longitude_to_vec(float colatitude,
                                                 float longitude) {
  float z = std::sin(M_PI / 2 - colatitude);
  float y_x_ratio = std::tan(longitude);
  // y / x = y_x_ratio
  // sqrt(z**2 + x**2 + y**2) = 1
  // y**2 + x**2 = 1 - z**2
  // y = y_x_ratio x
  // (y_x_ratio x)**2 + x**2 = 1 - z**2
  // (y_x_ratio x)**2 + x**2 = 1 - z**2
  // x = sqrt((1 - z**2) / (1 + y_x_ratio**2))
  // y = y_x_ratio x
  float x = std::sqrt((1 - z * z) / (1 + y_x_ratio * y_x_ratio));
  if (std::abs(longitude) > M_PI / 2) {
    x *= -1;
  }
  float y = x * y_x_ratio;

  return Eigen::Vector3f(x, y, z);
}

HOST_DEVICE inline std::tuple<float, float>
HalfSpherePartition::get_center_colatitude_longitude(unsigned collar,
                                                     unsigned region) const {
  // if collar is 0, we are in cap
  // else, we are in collar
  if (collar == 0) {
    return {0.0f, 0.0f};
  } else {
    float colatitude =
        (collar + 0.5f) / colatitude_inverse_interval_ - colatitude_offset_;

    const auto &c = colatitude_divs_[collar];
    assert(region + c.start_index < c.end_index);
    float longitude = ((region + 0.5f) / c.inverse_interval) - M_PI;

    return {colatitude, longitude};
  }
}

HOST_DEVICE inline Eigen::Vector3f
HalfSpherePartition::get_center_vec(unsigned collar, unsigned region) const {
  auto [colatitude, longitude] =
      get_center_colatitude_longitude(collar, region);

  return colatitude_longitude_to_vec(colatitude, longitude);
}

// colatitude should be 0 - pi
// longitude should be (-pi) - pi
HOST_DEVICE inline std::tuple<unsigned, bool>
HalfSpherePartition::get_closest(float colatitude, float longitude) const {
  const float half_pi = float(M_PI) / 2;
  bool is_flipped = colatitude > half_pi;
  if (colatitude > half_pi) {
    // flip to side of half
    colatitude = M_PI - colatitude;
    longitude = M_PI + longitude;
    if (longitude > M_PI) {
      longitude = -M_PI + (longitude - M_PI);
    }
  }

  // what epsilon?
  float epsilon = 1e-5;
  assert(colatitude < half_pi + epsilon);
  assert(-epsilon < colatitude);

  colatitude = std::clamp(colatitude, epsilon, half_pi - epsilon);

  assert(std::abs(longitude) < float(M_PI) + epsilon);

  longitude =
      std::clamp(longitude, float(-M_PI) + epsilon, float(M_PI) - epsilon);

  const auto &region = colatitude_divs_[std::floor(
      (colatitude + colatitude_offset_) * colatitude_inverse_interval_)];
  unsigned closest_idx =
      unsigned(std::floor(region.inverse_interval * (longitude + M_PI))) +
      region.start_index;

  assert(closest_idx < region.end_index);

  return {closest_idx, is_flipped};
}

// colatitude should be 0 - pi
// longitude should be (-pi) - pi
HOST_DEVICE inline std::tuple<unsigned, bool>
HalfSpherePartition::get_closest(const Eigen::Vector3f &vec) const {
  auto [colatitude, longitude] = vec_to_colatitude_longitude(vec);

  return get_closest(colatitude, longitude);
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
