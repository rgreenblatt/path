#pragma once

#include "lib/bgra.h"
#include "ray/execution_model.h"
#include "scene/scene.h"

#include <memory>

namespace ray {
template <ExecutionModel execution_model> class RendererImpl;

template <ExecutionModel execution_model> class Renderer {
public:
  Renderer(unsigned width, unsigned height, unsigned super_sampling_rate,
           unsigned recursive_iterations, std::unique_ptr<scene::Scene> &s);
  ~Renderer();
  void render(BGRA *pixels, const scene::Transform &m_film_to_world,
              const Eigen::Projective3f &world_to_film, bool use_kd_tree,
              bool show_times);

  scene::Scene &get_scene();

  const scene::Scene &get_scene() const;

private:
  // needs to not be smart pointer (compiler error otherwise)
  RendererImpl<execution_model> *renderer_impl_;
};
} // namespace ray
