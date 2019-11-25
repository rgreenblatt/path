#pragma once

#include "scene/scene.h"

#include <string>

namespace scene {
class CS123Scene : public Scene {
public:
  CS123Scene(const std::string &file_path, unsigned width, unsigned height);

  const Eigen::Affine3f &transform() {
    return transform_;
  }

private:
  Eigen::Affine3f transform_;
};
} // namespace scene
