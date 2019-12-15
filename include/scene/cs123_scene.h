#pragma once

#include "scene/scene.h"

#include <string>

namespace scene {
class CS123Scene : public Scene {
public:
  CS123Scene(const std::string &file_path, unsigned width, unsigned height);

  const Eigen::Affine3f &film_to_world() const { return film_to_world_; }

  const Eigen::Affine3f &world_to_film() const { return world_to_film_; }

  const Eigen::Projective3f &unhinging() const { return unhinging_; }

  void step(float) override {}

private:
  Eigen::Affine3f film_to_world_;
  Eigen::Affine3f world_to_film_;
  Eigen::Projective3f unhinging_;
};
} // namespace scene
