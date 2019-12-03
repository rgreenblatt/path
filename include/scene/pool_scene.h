#pragma once

#include "scene/scene.h"

#include <random>
#include <vector>

namespace scene {
class PoolScene : public Scene {
public:
  PoolScene();

  void step(float secs);
  
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
  // unit is 1"
  // pool ball is 2.25 in
  static constexpr float ball_radius = 2.25f / 2.0f;
  static constexpr float ball_center_y = 0.5f + ball_radius;
  static constexpr float wall_x_dist = 22.0f;
  static constexpr float wall_z_dist = 44.0f;
  static constexpr float sliding_friction = 5.0f;
  static constexpr float slide_to_roll = 20.0f;
  static constexpr float rolling_friction = 3.0f;

  TextureData loadTexture(const std::string &file);

  std::vector<BallState> states_;
  std::vector<unsigned> shuffles_;
  std::default_random_engine gen_;
};
} // namespace scene
