#include "scene/pool_scene.h"
#include "scene/texture_qimage.h"

#include <iostream>
#include <dbg.h>

namespace scene {
PoolScene::PoolScene() {
  float diffuse_coeff = 1.0;
  float specular_coeff = 1.0;
  float ambient_coeff = 0.3;
  
  std::string common_path = "images/";

  num_spheres_ = 0;
  num_cylinders_ = 0;
  num_cubes_ = 0;
  num_cones_ = 0;

  float ball_diffuse = 0.2;
  float ball_ambient = 0.5;
  float ball_reflective = 0.8;
  float ball_specular = 0.5;
  float ball_shininess = 25;

  std::string ball_path = common_path + "pool_ball_skins/";

  for (auto ball_data :
       {std::make_tuple(Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball1.jpg"),
        std::make_tuple(Eigen::Vector2f(3, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball2.jpg"),
        std::make_tuple(Eigen::Vector2f(4, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball3.jpg"),
        std::make_tuple(Eigen::Vector2f(5, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball4.jpg"),
        std::make_tuple(Eigen::Vector2f(6, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball5.jpg"),
        std::make_tuple(Eigen::Vector2f(7, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball6.jpg"),
        std::make_tuple(Eigen::Vector2f(8, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball7.jpg"),
        std::make_tuple(Eigen::Vector2f(9, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball8.jpg"),
        std::make_tuple(Eigen::Vector2f(10, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball9.jpg"),
        std::make_tuple(Eigen::Vector2f(11, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball10.jpg"),
        std::make_tuple(Eigen::Vector2f(12, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball11.jpg"),
        std::make_tuple(Eigen::Vector2f(13, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball12.jpg"),
        std::make_tuple(Eigen::Vector2f(14, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball13.jpg"),
        std::make_tuple(Eigen::Vector2f(15, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball14.jpg"),
        std::make_tuple(Eigen::Vector2f(16, -2), Eigen::Vector2f(2, -2),
                        Eigen::Vector2f(2, -2), Eigen::Vector2f(2, -2),
                        "Ball15.jpg")}) {
    auto [pos, vel, rot, angular_vel, filename] = ball_data;

    auto image_tex = load_qimage(ball_path + filename);

    if (!image_tex.has_value()) {
      std::cout << "couldn't load texture from file, exiting"
                << std::endl;
      std::exit(1);
    }
    
    unsigned texture_index = textures_.size();

    textures_.push_back(*image_tex);
    states_.push_back(BallState(shapes_.size(), pos, vel, rot, angular_vel));

    shapes_.push_back(ShapeData(
        Eigen::Affine3f::Identity(),
        Material(Color::Zero(), Color::Zero(),
                 Color::Ones() * ball_reflective * specular_coeff,
                 Color::Ones() * ball_specular * specular_coeff, Color::Zero(),
                 Color::Zero(), TextureData(texture_index, 1, 1),
                 ball_diffuse * diffuse_coeff, ball_ambient * ambient_coeff,
                 ball_shininess, 0)));

    num_spheres_++;
  }

  step(0.0f);

  auto light_color = Color(1.0, 1.0, 1.0);

  for (auto translate : {Eigen::Vector3f(5, 3, 5), Eigen::Vector3f(-4, 3, -4),
                         Eigen::Vector3f(3, 3, -3)}) {
    lights_.push_back(
        Light(light_color, PointLight(translate, Eigen::Array3f(1, 0.01, 0.005))));
  }

  copy_in_texture_refs();
}

void PoolScene::step(float secs) {
  for (auto &state : states_) {
    state.pos += state.vel * secs;
    state.rot += state.angular_vel * secs;

    shapes_[state.shape_index].set_transform(static_cast<Eigen::Affine3f>(
        Eigen::Translation3f(state.pos[0], ball_height, state.pos[0]) *
        Eigen::AngleAxis(state.rot[0], Eigen::Vector3f(0, 1, 0)) *
        Eigen::AngleAxis(state.rot[1], Eigen::Vector3f(1, 0, 0))));
  }
}
} // namespace scene
