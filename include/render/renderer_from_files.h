#pragma once

#include "execution_model/execution_model.h"
#include "lib/bgra_32.h"
#include "lib/float_rgb.h"
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
                  float width_height_ratio, bool quiet = false);
  void load_config(const std::filesystem::path &config_file_path);
  void print_config() const;

  double render(ExecutionModel execution_model, Span<BGRA32> pixels,
                unsigned samples_per, unsigned x_dim, unsigned y_dim,
                bool progress_bar = false, bool show_times = false);

  double render_float_rgb(ExecutionModel execution_model,
                          Span<FloatRGB> float_rgb, unsigned samples_per,
                          unsigned x_dim, unsigned y_dim,
                          bool progress_bar = false, bool show_times = false);

private:
  std::unique_ptr<Renderer> renderer_;
  std::unique_ptr<scene::Scene> scene_;
  std::unique_ptr<Settings> settings_;
};
} // namespace render
