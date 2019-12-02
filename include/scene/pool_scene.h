#pragma once

#include "scene/scene.h"

namespace scene {
class PoolScene : public Scene {
public:
  PoolScene();

  void step(float secs);

private:
  static constexpr float ball_height = 1.0f;
  static constexpr float ball_radius = 0.5f;
  static constexpr float wall_x_dist = 8.0f;
  static constexpr float wall_z_dist = 18.0f;

  struct BallState {
    unsigned shape_index;
    Eigen::Vector2f pos;
    Eigen::Vector2f vel;
    Eigen::Vector2f vel_change;
    Eigen::Vector2f rot;
    Eigen::Vector2f angular_vel;
    Eigen::Vector2f angular_vel_change;

    BallState(unsigned shape_index, Eigen::Vector2f pos, Eigen::Vector2f vel,
              Eigen::Vector2f rot, Eigen::Vector2f angular_vel)
        : shape_index(shape_index), pos(pos), vel(vel),
          vel_change(Eigen::Vector2f::Zero()), rot(rot),
          angular_vel(angular_vel),
          angular_vel_change(Eigen::Vector2f::Zero()) {}
  };


  TextureData loadTexture(const std::string &file);

  std::vector<BallState> states_;
};
} // namespace scene
