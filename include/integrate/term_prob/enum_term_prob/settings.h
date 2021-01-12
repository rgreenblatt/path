#pragma once

#include "integrate/term_prob/constant/settings.h"
#include "integrate/term_prob/enum_term_prob/term_prob_type.h"
#include "integrate/term_prob/multiplier_func/settings.h"
#include "integrate/term_prob/n_iters/settings.h"
#include "lib/settings.h"
#include "meta/pick_type.h"
#include "meta/predicate_for_all_values.h"

namespace integrate {
namespace term_prob {
namespace enum_term_prob {
template <TermProbType type>
struct Settings : public PickType<type, constant::Settings, n_iters::Settings,
                                  multiplier_func::Settings> {};

template <TermProbType type>
struct SettingsValid : BoolWrapper<Setting<Settings<type>>> {};

static_assert(PredicateForAllValues<TermProbType>::value<SettingsValid>);
} // namespace enum_term_prob
} // namespace term_prob
} // namespace integrate
