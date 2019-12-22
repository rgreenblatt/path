#include "ray/render.h"
#include "scene/camera.h"
#include "scene/pool_scene.h"
#include "scene/reflec_balls.h"

#include "ProgressBar.hpp"
#include <QImage>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <dbg.h>
#include <iostream>

template <ray::ExecutionModel execution_model>
void render_frames(unsigned width, unsigned height,
                   unsigned super_sampling_rate, scene::ReflecBalls &scene,
                   const Eigen::Affine3f &film_to_world,
                   const Eigen::Projective3f &world_to_film,
                   const std::string &dir_name, unsigned depth,
                   bool use_kd_tree, bool use_traversals,
                   bool use_traversal_dists, unsigned frames, float frame_rate,
                   unsigned physics_super_sampling_rate, bool make_video) {
#if 0
  if (!std::filesystem::exists(dir_name)) {
    std::filesystem::create_directories(dir_name);
  }
#endif
  std::unique_ptr<scene::Scene> s = std::make_unique<scene::ReflecBalls>(scene);

  ray::Renderer<execution_model> renderer(width, height, super_sampling_rate,
                                          depth, s);

  std::cout << "start:" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  float secs_per_frame = 1.0f / (frame_rate * physics_super_sampling_rate);

  ProgressBar progress(frames, 60);

  std::optional<cv::VideoWriter> writer;

  if (make_video) {
    writer =
        cv::VideoWriter(dir_name + "out.avi", CV_FOURCC('M', 'J', 'P', 'G'),
                        frame_rate, cv::Size(width, height));
  }

  cv::Mat frame(height, width, CV_8UC4);
  cv::Mat out(height, width, CV_8UC3);

  for (unsigned i = 0; i < frames; i++) {
    auto bgra_data = reinterpret_cast<BGRA *>(frame.data);

    renderer.render(bgra_data, film_to_world, world_to_film, use_kd_tree,
                    use_traversals, use_traversal_dists, false);

    for (unsigned i = 0; i < physics_super_sampling_rate; i++) {
      renderer.get_scene().step(secs_per_frame);
    }

    if (make_video) {
      cv::cvtColor(frame, out, CV_RGBA2RGB);
      writer->write(out);
    }

    ++progress;
    progress.display();
  }

  double time = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();

  std::cout << std::endl;
  std::cout << "rendered video in " << time << std::endl;
  std::cout << frames / time << " fps" << std::endl;
  std::cout << std::endl;

  if (make_video) {
    writer->release();
  }
}

int main(int argc, char *argv[]) {
  if (argc != 13) {
    std::cout << "wrong num args" << std::endl;

    return 1;
  }

  const unsigned depth = boost::lexical_cast<unsigned>(argv[1]);
  const unsigned width = boost::lexical_cast<unsigned>(argv[2]);
  const unsigned height = boost::lexical_cast<unsigned>(argv[3]);
  const unsigned super_sampling_rate = boost::lexical_cast<unsigned>(argv[4]);
  const bool use_kd_tree = boost::lexical_cast<bool>(argv[5]);
  const bool use_traversals = boost::lexical_cast<bool>(argv[6]);
  const bool use_traversal_dists = boost::lexical_cast<bool>(argv[7]);
  const bool render_cpu = boost::lexical_cast<bool>(argv[8]);
  const unsigned frames = boost::lexical_cast<unsigned>(argv[9]);
  const float frame_rate = boost::lexical_cast<float>(argv[10]);
  const unsigned physics_super_sampling_rate =
      boost::lexical_cast<unsigned>(argv[11]);
  const bool make_video = boost::lexical_cast<bool>(argv[12]);

  scene::ReflecBalls pool_scene;

  const std::string file_name = "out.png";

  auto [film_to_world, world_to_film] = scene::get_camera_transform(
      Eigen::Vector3f(-2, -1, 0), Eigen::Vector3f(0, 1, 0),
      Eigen::Vector3f(50, 30, 0), 1.0f, width, height, 30.0f);

  if (render_cpu) {
    std::cout << "rendering cpu" << std::endl;
    render_frames<ray::ExecutionModel::CPU>(
        width, height, super_sampling_rate, pool_scene, film_to_world,
        world_to_film, "cpu_video/", depth, use_kd_tree, use_traversals, 
        use_traversal_dists,
        frames,
        frame_rate, physics_super_sampling_rate, make_video);
    std::cout << "=============" << std::endl;
  }

  std::cout << "rendering gpu" << std::endl;
  render_frames<ray::ExecutionModel::GPU>(
      width, height, super_sampling_rate, pool_scene, film_to_world,
      world_to_film, "gpu_video/", depth, use_kd_tree, use_traversals,
      use_traversal_dists, frames, frame_rate, physics_super_sampling_rate,
      make_video);

  return 0;
}
