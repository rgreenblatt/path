#pragma once

#include "intersect/ray.h"

namespace intersect {
HOST_DEVICE inline Ray Ray::transform(const Eigen::Affine3f &transform) const {
  // TODO why where they using inverse().transpose()?
  return Ray{transform * origin, transform.linear() * direction};
}
} // namespace intersect
