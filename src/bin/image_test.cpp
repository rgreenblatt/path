#include "ray/render.h"
#include "scene/pool_scene.h"

int main(int argc, char *argv[]) { 
  scene::PoolScene scene;
  unsigned width = 100;
  unsigned height = 100;
  std::vector<BGRA> image_data(width * height);

  ray::render(scene, image_data.data(), width, height,
         static_cast<scene::Transform>(Eigen::Translation3f(0, 0, 2)));

  return 0;
}
