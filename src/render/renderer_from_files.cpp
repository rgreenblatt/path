#include "render/renderer_from_files.h"
#include "lib/assert.h"
#include "lib/serialize_enum.h"
#include "lib/serialize_tagged_union.h"
#include "meta/as_tuple/serialize.h"
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
                                   float width_height_ratio, bool quiet) {
  scene::scenefile_compat::ScenefileLoader loader;
  auto scene_op = loader.load_scene(scene_file_path, width_height_ratio, quiet);

  if (!scene_op.has_value()) {
    std::cerr << "failed to load scene" << std::endl;
    unreachable();
  }
  scene_ = std::make_unique<scene::Scene>(*scene_op);
}

void RendererFromFiles::load_config(
    const std::filesystem::path &config_file_path) {
  std::ifstream i(config_file_path);
  always_assert(!i.fail());
  always_assert(!i.bad());
  always_assert(i.is_open());
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
                               Span<BGRA32> pixels, unsigned samples_per,
                               unsigned x_dim, unsigned y_dim,
                               bool progress_bar, bool show_times) {
  renderer_->render(execution_model, pixels, *scene_, samples_per, x_dim, y_dim,
                    *settings_, progress_bar, show_times);
}

void RendererFromFiles::render_float_rgb(ExecutionModel execution_model,
                                         Span<FloatRGB> float_rgb,
                                         unsigned samples_per, unsigned x_dim,
                                         unsigned y_dim, bool progress_bar,
                                         bool show_times) {
  renderer_->render_float_rgb(execution_model, float_rgb, *scene_, samples_per,
                              x_dim, y_dim, *settings_, progress_bar,
                              show_times);
}
} // namespace render
