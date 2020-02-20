#pragma once

#include <Eigen/Core>

namespace render {
struct DirSample {
  Eigen::Vector3f direction;
  float prob;
};
} // namespace render
