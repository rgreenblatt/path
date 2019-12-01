#pragma once

#include "scene/scene.h"

namespace scene {
class PoolScene : public Scene {
public:
  PoolScene();

  void step(float secs);

private:
  static constexpr float ball_height = 1.0f;

  struct BallState {
    unsigned shape_index;
    Eigen::Vector2f pos;
    Eigen::Vector2f vel;
    Eigen::Vector2f rot;
    Eigen::Vector2f angular_vel;

    BallState(unsigned shape_index, Eigen::Vector2f pos, Eigen::Vector2f vel,
              Eigen::Vector2f rot, Eigen::Vector2f angular_vel)
        : shape_index(shape_index), pos(pos), vel(vel), rot(rot),
          angular_vel(angular_vel) {}
  };

  std::vector<BallState> states_;
};
} // namespace scene
