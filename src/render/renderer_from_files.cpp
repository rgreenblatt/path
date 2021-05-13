#include "render/renderer_from_files.h"

#include "render/config_io.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "scene/scenefile_compat/scenefile_loader.h"

#include <iostream>

namespace render {
RendererFromFiles::RendererFromFiles() {
  renderer_ = std::make_unique<Renderer>();
  settings_ = std::make_unique<Settings>();
}

RendererFromFiles::~RendererFromFiles() = default;
RendererFromFiles::RendererFromFiles(RendererFromFiles &&) = default;
RendererFromFiles &RendererFromFiles::operator=(RendererFromFiles &&) = default;

void RendererFromFiles::load_scene(const std::filesystem::path &scene_file_path,
                                   float width_height_ratio, bool quiet) {
  scene::scenefile_compat::ScenefileLoader loader;
  auto scene_op = loader.load_scene(scene_file_path, width_height_ratio, quiet);

  if (!scene_op.has_value()) {
    std::cerr << "failed to load scene" << std::endl;
    unreachable();
  }
  scene_camera_ =
      std::make_unique<scene::scenefile_compat::SceneCamera>(*scene_op);
}

void RendererFromFiles::load_config(
    const std::filesystem::path &config_file_path) {
  *settings_ = render::load_config(config_file_path);
}

void RendererFromFiles::print_config() const {
  render::print_config(*settings_);
}

double RendererFromFiles::render(ExecutionModel execution_model, unsigned x_dim,
                                 unsigned y_dim, const Output &output,
                                 unsigned samples_per, bool show_progress,
                                 bool show_times) {
  return renderer_->render(execution_model,
                           {tag_v<SampleSpecType::SquareImage>,
                            {.x_dim = x_dim,
                             .y_dim = y_dim,
                             .film_to_world = scene_camera_->film_to_world}},
                           output, scene_camera_->scene, samples_per,
                           *settings_, show_progress, show_times);
}
} // namespace render
