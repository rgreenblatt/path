#pragma once

#include "integrate/dir_sampler/enum_dir_sampler/dir_sampler_type.h"
#include "integrate/light_sampler/enum_light_sampler/light_sampler_type.h"
#include "integrate/term_prob/enum_term_prob/term_prob_type.h"
#include "intersect/accel/enum_accel/accel_type.h"
#include "lib/attribute.h"
#include "rng/enum_rng/rng_type.h"

#include <compare>

namespace render {
namespace enum_accel = intersect::accel::enum_accel;
namespace enum_dir_sampler = integrate::dir_sampler::enum_dir_sampler;
namespace enum_light_sampler = integrate::light_sampler::enum_light_sampler;
namespace enum_term_prob = integrate::term_prob::enum_term_prob;

using enum_accel::AccelType;
using enum_dir_sampler::DirSamplerType;
using enum_light_sampler::LightSamplerType;
using enum_term_prob::TermProbType;
using rng::enum_rng::RngType;

struct CompileTimeSettings {
  AccelType flat_accel_type;
  LightSamplerType light_sampler_type;
  DirSamplerType dir_sampler_type;
  TermProbType term_prob_type;
  RngType rng_type;

  ATTR_PURE constexpr auto
  operator<=>(const CompileTimeSettings &other) const = default;
};
} // namespace render
