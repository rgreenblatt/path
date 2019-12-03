#include "scene/pool_scene.h"
#include "scene/texture_qimage.h"

#include <boost/iterator/counting_iterator.hpp>

#include <chrono>
#include <dbg.h>
#include <iostream>

namespace scene {
TextureData PoolScene::loadTexture(const std::string &file) {
  auto image_tex = load_qimage(file);

  if (!image_tex.has_value()) {
    std::cout << "couldn't load texture from file, exiting" << std::endl;
    std::exit(1);
  }

  unsigned texture_index = textures_.size();

  textures_.push_back(*image_tex);

  return TextureData(texture_index, 1, 1);
}

void PoolScene::setBreak() {
  float max_noise = 1e-1;
  std::uniform_real_distribution<float> dist(-max_noise, max_noise);
  // TODO:
  //  - measure cue ball starting position
  //  - measure break point
  //  - noise
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
}

// units are 1"
PoolScene::PoolScene() {

  gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
  float diffuse_coeff = 0.5;
  float specular_coeff = 1.0;
  float ambient_coeff = 0.4;

  std::string common_path = "images/";

  num_spheres_ = 0;
  num_cylinders_ = 0;
  num_cubes_ = 0;
  num_cones_ = 0;

  float ball_diffuse = 0.5;
  float ball_ambient = 0.5;
  float ball_reflective = 0.2;
  float ball_specular = 0.5;
  float ball_shininess = 25;

  std::string ball_path = common_path + "pool_ball_skins/";

  for (auto filename :
       {"BallCue.jpg", "Ball1.jpg", "Ball2.jpg", "Ball3.jpg", "Ball4.jpg",
        "Ball5.jpg", "Ball6.jpg", "Ball7.jpg", "Ball8.jpg", "Ball9.jpg",
        "Ball10.jpg", "Ball11.jpg", "Ball12.jpg", "Ball13.jpg", "Ball14.jpg",
        "Ball15.jpg"}) {
    auto texture_data = loadTexture(ball_path + filename);

    states_.push_back(BallState(shapes_.size()));

    shapes_.push_back(ShapeData(
        Eigen::Affine3f::Identity(),
        Material(Color::Zero(), Color::Zero(),
                 Color::Ones() * ball_reflective * specular_coeff,
                 Color::Ones() * ball_specular * specular_coeff, Color::Zero(),
                 Color::Zero(), texture_data, ball_diffuse * diffuse_coeff,
                 ball_ambient * ambient_coeff, ball_shininess, 0)));

    num_spheres_++;
  }

  shuffles_ = std::vector(
      boost::make_counting_iterator(static_cast<unsigned>(0)),
      boost::make_counting_iterator(static_cast<unsigned>(states_.size())));

  setBreak();

  float surface_diffuse = 0.5;
  float surface_ambient = 0.5;

  auto surface_texture_data =
      loadTexture(common_path + "pool_table_surface.jpg");

  shapes_.push_back(ShapeData(
      static_cast<Eigen::Affine3f>(
          Eigen::Scaling(Eigen::Vector3f(wall_x_dist * 2, 1, wall_z_dist * 2))),
      Material(Color::Zero(), Color::Zero(), Color::Zero(), Color::Zero(),
               Color::Zero(), Color::Zero(), surface_texture_data,
               surface_diffuse * diffuse_coeff, surface_ambient * ambient_coeff,
               0, 0)));

  num_cubes_++;

  float side_width = 1;
  float wall_height = 5.5;

  Eigen::Translation3f x_translate(wall_x_dist + side_width / 2, 0, 0);
  Eigen::Translation3f z_translate(0, 0, wall_z_dist + side_width / 2);

  auto z_scale =
      Eigen::Scaling(Eigen::Vector3f(side_width, wall_height, wall_z_dist * 2));
  auto x_scale = Eigen::Scaling(Eigen::Vector3f(
      wall_x_dist * 2 + 2 * side_width, wall_height, side_width));

  BGRA bgra_color(48, 38, 31, 0);

  Color color = bgra_color.head<3>().cast<float>() / 255.0f;

  for (const auto &transform :
       {x_translate * z_scale, x_translate.inverse() * z_scale,
        z_translate * x_scale, z_translate.inverse() * x_scale}) {
    shapes_.push_back(ShapeData(
        transform, Material(color * diffuse_coeff, color * ambient_coeff,
                            Color::Zero(), Color::Zero(), Color::Zero(),
                            Color::Zero(), thrust::nullopt, 0, 0, 25, 0)));

    num_cubes_++;
  }

  step(0.0f);

  auto light_color = Color(1.0, 1.0, 1.0);

  for (auto translate : {
           Eigen::Vector3f(0, 8, 30),
           Eigen::Vector3f(0, 8, -30),
       }) {
    lights_.push_back(Light(
        light_color, PointLight(translate, Eigen::Array3f(1, 0.0, 0.001))));
  }

  for (auto dir : {Eigen::Vector3f(-1, -1, 0), Eigen::Vector3f(1, -1, 0)}) {
    lights_.push_back(Light(light_color * 0.5, DirectionalLight(dir)));
  }

  copyInTextureRefs();
}

void PoolScene::step(float secs) {
  // static to avoid allocations...(not sure if needed...)
  for (unsigned state_index = 0; state_index < states_.size(); state_index++) {
    auto &state = states_[state_index];

#if 1
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
#endif

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

    shapes_[state.shape_index].set_transform(static_cast<Eigen::Affine3f>(
        Eigen::Translation3f(state.pos[0], ball_center_y, state.pos[1]) *
        state.rot * Eigen::Scaling(ball_scale, ball_scale, ball_scale)));
  }
}
} // namespace scene
