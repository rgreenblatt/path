#include "render/renderer_from_files.h"
#include "lib/serialize_enum.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "scene/scenefile_compat/scenefile_loader.h"

#include <cereal-yaml/archives/yaml.hpp>

#include <fstream>
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
                                   float width_height_ratio) {
  scene::scenefile_compat::ScenefileLoader loader;
  auto scene_op = loader.load_scene(scene_file_path, width_height_ratio);

  if (!scene_op.has_value()) {
    std::cerr << "failed to load scene" << std::endl;
    assert(false);
    abort();
  }
  scene_ = std::make_unique<scene::Scene>(*scene_op);
}

void RendererFromFiles::load_config(
    const std::filesystem::path &config_file_path) {
  std::ifstream i(config_file_path);
  cereal::YAMLInputArchive archive(i);
  archive(cereal::make_nvp("settings", *settings_));
}

void RendererFromFiles::print_config() const {
  std::ostringstream os;
  {
    cereal::YAMLOutputArchive archive(os);
    archive(cereal::make_nvp("settings", *settings_));
  }
  std::cout << os.str() << std::endl;
}

void RendererFromFiles::render(ExecutionModel execution_model,
                               Span<BGRA> pixels, unsigned &samples_per,
                               unsigned x_dim, unsigned y_dim,
                               bool progress_bar, bool show_times) {
  renderer_->render(execution_model, pixels, *scene_, samples_per, x_dim, y_dim,
                    *settings_, progress_bar, show_times);
}

void RendererFromFiles::render_intensities(ExecutionModel execution_model,
                                           Span<Eigen::Array3f> intensities,
                                           unsigned &samples_per,
                                           unsigned x_dim, unsigned y_dim,
                                           bool progress_bar, bool show_times) {
  renderer_->render_intensities(execution_model, intensities, *scene_,
                                samples_per, x_dim, y_dim, *settings_,
                                progress_bar, show_times);
}
} // namespace render
