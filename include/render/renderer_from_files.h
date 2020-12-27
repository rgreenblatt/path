#pragma once

#include "execution_model/execution_model.h"
#include "lib/bgra.h"
#include "lib/span.h"

#include <filesystem>
#include <memory>

namespace scene {
class Scene;
}

namespace render {
class Renderer;
struct Settings;

class RendererFromFiles {
public:
  // need to implementated when Impl is defined
  RendererFromFiles();
  ~RendererFromFiles();
  RendererFromFiles(RendererFromFiles &&);
  RendererFromFiles &operator=(RendererFromFiles &&);

  void load_scene(const std::filesystem::path &scene_file_path,
                  float width_height_ratio);
  void load_config(const std::filesystem::path &config_file_path);
  void print_config() const;

  void render(ExecutionModel execution_model, Span<BGRA> pixels,
              unsigned &samples_per, unsigned x_dim, unsigned y_dim,
              bool progress_bar = false, bool show_times = false);

  void render_intensities(ExecutionModel execution_model,
                          Span<Eigen::Array3f> intensities,
                          unsigned &samples_per, unsigned x_dim, unsigned y_dim,
                          bool progress_bar = false, bool show_times = false);

private:
  std::unique_ptr<Renderer> renderer_;
  std::unique_ptr<scene::Scene> scene_;
  std::unique_ptr<Settings> settings_;
};
} // namespace render
