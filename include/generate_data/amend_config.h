#pragma once

#include "render/settings.h"

namespace generate_data {
inline void amend_config(render::Settings &settings) {
  settings.term_prob = {
      tag_v<integrate::term_prob::enum_term_prob::TermProbType::NIters>,
      integrate::term_prob::n_iters::Settings{.iters = 0}};
  settings.rendering_equation_settings = {
      .back_cull_emission = false,
  };
}
} // namespace generate_data
