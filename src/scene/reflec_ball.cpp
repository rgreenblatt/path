#include "scene/reflec_balls.h"

#include <boost/iterator/counting_iterator.hpp>

#include <chrono>

namespace scene {
void ReflecBalls::setBreak() {
#if 0
  float max_noise = 1e-1;
  std::uniform_real_distribution<float> dist(-max_noise, max_noise);
  for (auto &ball : states_) {
    ball.angular_vel = Eigen::Vector3f::Zero();
    ball.rot = Eigen::Quaternionf::Identity();
    ball.vel = Eigen::Vector2f::Zero();
  }

  auto &cue_ball = states_[0];
  cue_ball.pos = Eigen::Vector2f(0, 25);

  float start_z = -10.0f;
  float z_per_row = std::sqrt(3.0f) * ball_radius + 2e-1;

  std::shuffle(shuffles_.begin(), shuffles_.end(), gen_);

  unsigned shuffle_index = 0;
  for (unsigned row = 0; row < 5; row++) {
    float start_x = row * (ball_radius + 2e-1);
    for (unsigned col = 0; col < row + 1; col++) {
      unsigned ball_index = 0;
      if (row == 2 && col == 1) {
        ball_index = 8;
      } else {
        do {
          ball_index = shuffles_[shuffle_index];
          shuffle_index++;
        } while (ball_index == 8 || ball_index == 0);
      }
      states_[ball_index].pos =
          Eigen::Vector2f(start_x - col * (ball_radius * 2 + 2e-1) + dist(gen_),
                          start_z - row * z_per_row + dist(gen_));
      ball_index++;
    }
  }
#endif
}

// units are 1"
ReflecBalls::ReflecBalls() {
  gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
  float diffuse_coeff = 0.5;
  float specular_coeff = 1.0;
  float ambient_coeff = 0.4;

  std::string common_path = "images/";

  float ball_diffuse = 0.5;
  float ball_ambient = 0.5;
  float ball_reflective = 0.2;
  float ball_specular = 0.5;
  float ball_shininess = 25;

  std::string ball_path = common_path + "pool_ball_skins/";

#if 0
  for (auto filename :
       {"BallCue.jpg", "Ball1.jpg", "Ball2.jpg", "Ball3.jpg", "Ball4.jpg",
        "Ball5.jpg", "Ball6.jpg", "Ball7.jpg", "Ball8.jpg", "Ball9.jpg",
        "Ball10.jpg", "Ball11.jpg", "Ball12.jpg", "Ball13.jpg", "Ball14.jpg",
        "Ball15.jpg"}) {
    auto texture_data = loadTexture(ball_path + filename);

    states_.push_back(BallState(addShape(ShapeData(
        Eigen::Affine3f::Identity(),
        Material(Color::Zero(), Color::Zero(),
                 Color::Ones() * ball_reflective * specular_coeff,
                 Color::Ones() * ball_specular * specular_coeff, Color::Zero(),
                 Color::Zero(), texture_data, ball_diffuse * diffuse_coeff,
                 ball_ambient * ambient_coeff, ball_shininess, 0),
        scene::Shape::Sphere))));
  }
#endif

  shuffles_ = std::vector(
      boost::make_counting_iterator(static_cast<unsigned>(0)),
      boost::make_counting_iterator(static_cast<unsigned>(states_.size())));

  setBreak();

  const std::array<BGRA, 6> colors = {{
      {60, 60, 60, 0},
      {60, 60, 60, 0},
      {168, 159, 96, 0},
      {96, 159, 168, 0},
      {60, 60, 60, 0},
      {60, 60, 60, 0},
  }};

  for (std::tuple<int, Eigen::AngleAxis<float>> axis_rot : {
           std::make_tuple(
               0, Eigen::AngleAxis(float(M_PI / 2), Eigen::Vector3f(0, 0, 1))),
           std::make_tuple(1, Eigen::AngleAxis(0.0f, Eigen::Vector3f(0, 1, 0))),
           std::make_tuple(
               2, Eigen::AngleAxis(float(M_PI / 2), Eigen::Vector3f(1, 0, 0))),
       }) {
    static constexpr float surface_dim = 100.0f;
    static constexpr float surface_width = 8.0f;
    static constexpr float surface_gap = 30.0f;
    static constexpr float surface_inside_dist = 10.0f;

    for (bool neg : {false, true}) {
      auto [axis, rot] = axis_rot;
      float surface_diffuse = 0.5;
      float surface_ambient = 0.5;
      const auto &color =
          colors[axis * 2 + neg].head<3>().cast<float>() / 255.0f;
      Eigen::Vector3f axis_vec = Eigen::Vector3f::Zero();
      axis_vec[axis] = surface_dim + surface_gap;
      if (axis == 1) {
        axis_vec[axis] -= surface_inside_dist + surface_gap;
      }
      if (neg) {
        axis_vec[axis] *= -1.0f;
      }
      addShape(ShapeData(
          static_cast<Eigen::Affine3f>(
              Eigen::Translation3f(axis_vec) * rot *
              Eigen::Scaling(Eigen::Vector3f(surface_dim * 2, surface_width,
                                             surface_dim * 2))),
          Material(color * diffuse_coeff, color * ambient_coeff,
                   color * specular_coeff, color * specular_coeff,
                   Color::Zero(), Color::Zero(), thrust::nullopt, 0, 0, 25, 0),
          scene::Shape::Cube));
    }
  }

  step(0.0f);

  auto light_color = Color(1.0, 1.0, 1.0);

  for (auto translate : {
           Eigen::Vector3f(0, 8, 1000),
           /* Eigen::Vector3f(0, 8, -30), */
       }) {
    addLight(Light(light_color,
                   PointLight(translate, Eigen::Array3f(1, 0.0, 0.001))));
  }

#if 0
  for (auto dir : {Eigen::Vector3f(-1, -1, 0), Eigen::Vector3f(1, -1, 0)}) {
    addLight(Light(light_color * 0.5, DirectionalLight(dir)));
  }
#endif

  finishConstructScene();
}

void ReflecBalls::step(float secs) {
#if 0
  // static to avoid allocations...(not sure if needed...)
  for (unsigned state_index = 0; state_index < states_.size(); state_index++) {
    auto &state = states_[state_index];

    Eigen::Vector2f vel_due_to_spin(-state.angular_vel.z() * ball_radius,
                                    state.angular_vel.x() * ball_radius);
    auto slip_rate = (state.vel - vel_due_to_spin).eval();
    float slip_rate_norm = slip_rate.norm();
    if (slip_rate_norm < 5e-3f) {
      float vel_norm = state.vel.norm();
      if (vel_norm < 5e-5f) {
        state.vel = Eigen::Vector2f::Zero();
      } else {
        // different coeff??? TODO
        state.vel -= state.vel * sliding_friction * secs / vel_norm;
      }
    } else {
      state.vel -= slip_rate * sliding_friction * secs / slip_rate_norm;
      auto change_angular =
          (slip_rate * slide_to_roll * secs / slip_rate_norm / ball_radius)
              .eval();
      state.angular_vel.z() -= change_angular[0];
      state.angular_vel.x() += change_angular[1];
    }
    float angular_vel_norm = state.angular_vel.norm();
    if (angular_vel_norm < 1e-7f) {
      state.angular_vel = Eigen::Vector3f::Zero();
    } else {
      state.angular_vel -=
          rolling_friction * secs * state.angular_vel / angular_vel_norm;
    }

    auto new_pos = (state.pos + state.vel * secs).eval();
    auto angular_vel_time = state.angular_vel * secs;
    Eigen::Quaternionf new_rot =
        Eigen::AngleAxis(angular_vel_time.x(), Eigen::Vector3f(1, 0, 0)) *
        Eigen::AngleAxis(angular_vel_time.y(), Eigen::Vector3f(0, 1, 0)) *
        Eigen::AngleAxis(angular_vel_time.z(), Eigen::Vector3f(0, 0, 1)) *
        state.rot;

    std::shuffle(shuffles_.begin(), shuffles_.end(), gen_);

    // collisions
    for (unsigned other_state_index : shuffles_) {
      if (other_state_index <= state_index) {
        continue;
      }

      auto &other_state = states_[other_state_index];
      auto other_new_pos = (other_state.pos + other_state.vel * secs).eval();

      auto pos_diff = (new_pos - other_new_pos).eval();
      auto pos_diff_norm = pos_diff.norm();

      if (pos_diff_norm < 2 * ball_radius - 1e-4f) {
        // not quite correct...
        auto change = (((state.vel - other_state.vel).dot(pos_diff) /
                        pos_diff.squaredNorm()) *
                       pos_diff)
                          .eval();
        state.vel_change -= change;
        other_state.vel_change += change;
      }
    }

    if (std::abs(new_pos[0]) > wall_x_dist - ball_radius) {
      state.vel_change -= Eigen::Vector2f(2.0f * state.vel[0], 0);
    }

    if (std::abs(new_pos[1]) > wall_z_dist - ball_radius) {
      state.vel_change -= Eigen::Vector2f(0, 2.0f * state.vel[1]);
    }

    state.pos = new_pos;
    state.rot = new_rot;

    state.vel += state.vel_change;
    state.vel_change.setZero();

    float ball_scale = ball_radius * 2.0f;

    updateTransformShape(
        state.shape_index,
        static_cast<Eigen::Affine3f>(
            Eigen::Translation3f(state.pos[0], ball_center_y, state.pos[1]) *
            state.rot * Eigen::Scaling(ball_scale, ball_scale, ball_scale)));
  }
#endif
}
} // namespace scene