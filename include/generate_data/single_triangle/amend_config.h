#pragma once

#include "render/settings.h"

namespace generate_data {
namespace single_triangle {
static void amend_config(render::Settings &settings) {
  // TODO: is sobel fine???
  settings.term_prob = {
      tag_v<integrate::term_prob::enum_term_prob::TermProbType::NIters>,
      integrate::term_prob::n_iters::Settings{.iters = 0}};
  settings.rendering_equation_settings = {
      .back_cull_emission = true,
  };
}
} // namespace single_triangle
} // namespace generate_data
