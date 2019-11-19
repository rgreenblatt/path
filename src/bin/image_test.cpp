#include "ray/render.h"
#include "scene/pool_scene.h"

#include <QImage>

#include <chrono>
#include <iostream>

int main(int argc, char *argv[]) {
  scene::PoolScene scene;
  unsigned width = 1920;
  unsigned height = 1080;

  QImage image(width, height, QImage::Format_RGB32);

  auto bgra_data = reinterpret_cast<BGRA *>(image.bits());

  ray::Renderer renderer(width, height);

  renderer.render(scene, bgra_data,
                  static_cast<scene::Transform>(Eigen::Translation3f(0, 0, 9)));

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++) {
    renderer.render(
        scene, bgra_data,
        static_cast<scene::Transform>(Eigen::Translation3f(0, 0, 9)));
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "render time:"
            << std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                         start)
                   .count()
            << std::endl;

  image.save("out.png");

  return 0;
}
