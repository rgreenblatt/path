#include "scene/pool_scene.h"
#include "scene/texture_qimage.h"

#include <boost/range/combine.hpp>

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

PoolScene::PoolScene() {
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

  for (auto ball_data : {
         std::make_tuple(Eigen::Vector2f(-6, -2), Eigen::Vector2f(1, -2),
                         Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                         "BallCue.jpg"),
#if 1
             std::make_tuple(Eigen::Vector2f(-4, -2), Eigen::Vector2f(-1, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball1.jpg"),
             std::make_tuple(Eigen::Vector2f(-2, -2), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball2.jpg"),
             std::make_tuple(Eigen::Vector2f(0, -2), Eigen::Vector2f(2, -3),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball3.jpg"),
             std::make_tuple(Eigen::Vector2f(2, -2), Eigen::Vector2f(1.3, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball4.jpg"),
             std::make_tuple(Eigen::Vector2f(4, -2), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball5.jpg"),
             std::make_tuple(Eigen::Vector2f(6, -2), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball6.jpg"),
             std::make_tuple(Eigen::Vector2f(-6, -4), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball7.jpg"),
             std::make_tuple(Eigen::Vector2f(-4, -4), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball8.jpg"),
             std::make_tuple(Eigen::Vector2f(-2, -4), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball9.jpg"),
             std::make_tuple(Eigen::Vector2f(0, -4), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball10.jpg"),
             std::make_tuple(Eigen::Vector2f(2, -4), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball11.jpg"),
             std::make_tuple(Eigen::Vector2f(4, -4), Eigen::Vector2f(2, -4),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball12.jpg"),
             std::make_tuple(Eigen::Vector2f(6, -4), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball13.jpg"),
             std::make_tuple(Eigen::Vector2f(2, -6), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball14.jpg"),
#endif
             std::make_tuple(Eigen::Vector2f(4, -6), Eigen::Vector2f(2, -2),
                             Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                             "Ball15.jpg")
       }) {
    auto [pos, vel, rot, angular_vel, filename] = ball_data;

    auto texture_data = loadTexture(ball_path + filename);

    states_.push_back(BallState(shapes_.size(), pos, vel, rot, angular_vel));

    shapes_.push_back(ShapeData(
        Eigen::Affine3f::Identity(),
        Material(Color::Zero(), Color::Zero(),
                 Color::Ones() * ball_reflective * specular_coeff,
                 Color::Ones() * ball_specular * specular_coeff, Color::Zero(),
                 Color::Zero(), texture_data, ball_diffuse * diffuse_coeff,
                 ball_ambient * ambient_coeff, ball_shininess, 0)));

    num_spheres_++;
  }

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
  float wall_height = 8.5;

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
        transform,
        Material(color * diffuse_coeff, color * ambient_coeff,
                 color * specular_coeff, color * specular_coeff, Color::Zero(),
                 Color::Zero(), thrust::nullopt, 0, 0, 25, 0)));

    num_cubes_++;
  }

  step(0.0f);

  auto light_color = Color(1.0, 1.0, 1.0);

  for (auto translate : {
           Eigen::Vector3f(0, 3, 12),
           Eigen::Vector3f(0, 3, 0),
           Eigen::Vector3f(0, 3, -12),
       }) {
    lights_.push_back(Light(
        light_color, PointLight(translate, Eigen::Array3f(1, 0.0, 0.01))));
  }

  for (auto dir : {Eigen::Vector3f(-1, -1, 0), Eigen::Vector3f(1, -1, 0)}) {
    lights_.push_back(Light(light_color * 0.5, DirectionalLight(dir)));
  }

  copy_in_texture_refs();
}

void PoolScene::step(float secs) {
  for (unsigned state_index = 0; state_index < states_.size(); state_index++) {
    auto &state = states_[state_index];
    auto new_pos = (state.pos + state.vel * secs).eval();
    auto new_rot = (state.rot + state.angular_vel * secs).eval();
    for (unsigned other_state_index = state_index + 1;
         other_state_index < states_.size(); other_state_index++) {
      auto &other_state = states_[other_state_index];
      auto other_new_pos = (other_state.pos + other_state.vel * secs).eval();

      if ((other_new_pos - new_pos).norm() < 2 * ball_radius - 1e-3) {
        // not quite correct...
        auto pos_diff = (state.pos - other_state.pos).eval();
        auto change =
            (((state.vel - other_state.vel).dot(pos_diff) / pos_diff.norm()) *
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
    state.vel += state.vel_change;
    state.vel_change.setZero();
    state.rot = new_rot;

    shapes_[state.shape_index].set_transform(static_cast<Eigen::Affine3f>(
        Eigen::Translation3f(state.pos[0], ball_height, state.pos[1]) *
        Eigen::AngleAxis(state.rot[0], Eigen::Vector3f(0, 1, 0)) *
        Eigen::AngleAxis(state.rot[1], Eigen::Vector3f(1, 0, 0))));
  }
}
} // namespace scene
