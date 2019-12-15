#pragma once

#include "scene/scene.h"

#include <random>
#include <vector>

namespace scene {
class ReflecBalls : public Scene {
public:
  ReflecBalls();

  void step(float secs) override;

  void setBreak();

  struct BallState {
    unsigned shape_index;
    Eigen::Vector2f pos;
    Eigen::Vector2f vel;
    Eigen::Vector2f vel_change;
    Eigen::Quaternionf rot;
    Eigen::Vector3f angular_vel;
    Eigen::Vector3f angular_vel_change;

    BallState(unsigned shape_index)
        : shape_index(shape_index), vel_change(Eigen::Vector2f::Zero()),
          angular_vel_change(Eigen::Vector3f::Zero()) {}
  };

  BallState &getBallState(unsigned index) { return states_[index]; }

  unsigned getNumBalls() { return states_.size(); }

private:
  static constexpr float ball_radius = 1.0f;
  static constexpr float sliding_friction = 5.0f;
  static constexpr float slide_to_roll = 20.0f;
  static constexpr float rolling_friction = 3.0f;

  std::vector<BallState> states_;
  std::vector<unsigned> shuffles_;
  std::default_random_engine gen_;
};
} // namespace scene
