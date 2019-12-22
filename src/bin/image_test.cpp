#include "ray/render.h"
#include "scene/cs123_scene.h"
#include "scene/pool_scene.h"

#include <QImage>
#include <boost/lexical_cast.hpp>

#include <chrono>
#include <iostream>

template <ray::ExecutionModel execution_model>
void run_test(unsigned width, unsigned height, unsigned super_sampling_rate,
              const scene::CS123Scene &scene,
              const Eigen::Affine3f &film_to_world,
              const Eigen::Projective3f &world_to_film,
              const std::string &filename, unsigned depth, bool use_kd_tree,
              bool use_traversals, bool use_traversal_dists) {
  QImage image(width, height, QImage::Format_RGB32);

  auto bgra_data = reinterpret_cast<BGRA *>(image.bits());

  std::unique_ptr<scene::Scene> s = std::make_unique<scene::CS123Scene>(scene);

  ray::Renderer<execution_model> renderer(width, height, super_sampling_rate,
                                          depth, s);

#if 1
  // realistic memory benchmark
  renderer.render(bgra_data, film_to_world, world_to_film, use_kd_tree,
                  use_traversals, use_traversal_dists, false);
#endif

  std::cout << "start:" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  renderer.render(bgra_data, film_to_world, world_to_film, use_kd_tree,
                  use_traversals, use_traversal_dists, true);

#if 1
  unsigned x_rad = 3;
  unsigned y_rad = 3;

  auto draw_point = [&](unsigned x, unsigned y, BGRA color) {
    for (unsigned y_draw = std::max(y, y_rad) - y_rad;
         y_draw <= std::min(y, height - y_rad); y_draw++) {
      for (unsigned x_draw = std::max(x, x_rad) - x_rad;
           x_draw <= std::min(x, width - x_rad); x_draw++) {
        bgra_data[x_draw + y_draw * width] = color;
      }
    }
  };

  draw_point(570, 260, BGRA(100, 100, 255, 0));
#endif

  std::cout << "rendered in "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count()
            << std::endl;

  image.save(filename.c_str());
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    std::cout << "wrong num args" << std::endl;

    return 1;
  }

  const unsigned depth = boost::lexical_cast<unsigned>(argv[2]);
  const unsigned width = boost::lexical_cast<unsigned>(argv[3]);
  const unsigned height = boost::lexical_cast<unsigned>(argv[4]);
  const unsigned super_sampling_rate = boost::lexical_cast<unsigned>(argv[5]);
  const bool use_kd_tree = boost::lexical_cast<bool>(argv[6]);
  const bool use_traversals = boost::lexical_cast<bool>(argv[7]);
  const bool use_traversal_dists = boost::lexical_cast<bool>(argv[8]);
  const bool render_cpu = boost::lexical_cast<bool>(argv[9]);

  scene::CS123Scene scene(argv[1], width, height);

  const std::string file_name = "out.png";

  /* auto transform = static_cast<Eigen::Affine3f>(Eigen::Translation3f(7, 0,
   * 10)); */
  auto film_to_world = scene.film_to_world();
  auto world_to_film = scene.world_to_film();

  if (render_cpu) {
    std::cout << "rendering cpu" << std::endl;
    run_test<ray::ExecutionModel::CPU>(width, height, super_sampling_rate,
                                       scene, film_to_world, world_to_film,
                                       "cpu_" + file_name, depth, use_kd_tree,
                                       use_traversals, use_traversal_dists);
    std::cout << "=============" << std::endl;
  }

  std::cout << "rendering gpu" << std::endl;
  run_test<ray::ExecutionModel::GPU>(width, height, super_sampling_rate, scene,
                                     film_to_world, world_to_film,
                                     "gpu_" + file_name, depth, use_kd_tree,
                                     use_traversals, use_traversal_dists);

  return 0;
}
