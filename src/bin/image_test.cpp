#include "ray/render.h"
#include "scene/cs123_scene.h"

#include <QImage>
#include <boost/lexical_cast.hpp>

#include <chrono>
#include <iostream>

#include <dbg.h>

int main(int argc, char *argv[]) {
  if (argc <= 1) {
    std::cout << "too few args" << std::endl;

    return 1;
  }

  for (size_t i = 1; i < static_cast<size_t>(argc); i++) {
    scene::CS123Scene scene(argv[i]);
    unsigned width = 1000;
    unsigned height = 1000;

    QImage image(width, height, QImage::Format_RGB32);

    auto bgra_data = reinterpret_cast<BGRA *>(image.bits());

    ray::Renderer<ray::ExecutionModel::CPU> renderer(width, height, 1);

    const Eigen::Affine3f transform(Eigen::Translation3f(0, 0, 5));
    std::cout << transform.matrix() << std::endl;

    renderer.render(
        scene, bgra_data,
        static_cast<scene::Transform>(transform));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
      renderer.render(
          scene, bgra_data,
          static_cast<scene::Transform>(transform));
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "render time:"
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     end - start)
                     .count()
              << std::endl;

    image.save(("out_" + boost::lexical_cast<std::string>(i) + ".png").c_str());
  }

  return 0;
}
