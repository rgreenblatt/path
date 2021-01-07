#pragma once

#include "integrate/dir_sampler/dir_sampler.h"
#include "integrate/light_sampler/light_sampler.h"
#include "integrate/rendering_equation_state.h"
#include "integrate/term_prob/term_prob.h"
#include "intersectable_scene/scene_ref.h"

namespace integrate {
// helper struct to package up some of the inputs
template <intersectable_scene::SceneRef S,
          light_sampler::LightSamplerRef<typename S::B> LIn,
          dir_sampler::DirSamplerRef<typename S::B> D, term_prob::TermProbRef T>
struct RenderingEquationComponents {
  using L = LIn;
  using B = typename S::B;
  using InfoType = typename S::InfoType;

  [[no_unique_address]] S scene;
  [[no_unique_address]] L light_sampler;
  [[no_unique_address]] D dir_sampler;
  [[no_unique_address]] T term_prob;
};
} // namespace integrate
