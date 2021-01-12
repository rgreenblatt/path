#pragma once

#include "integrate/term_prob/constant/constant.h"
#include "integrate/term_prob/enum_term_prob/settings.h"
#include "integrate/term_prob/enum_term_prob/term_prob_type.h"
#include "integrate/term_prob/multiplier_func/multiplier_func.h"
#include "integrate/term_prob/n_iters/n_iters.h"
#include "integrate/term_prob/term_prob.h"
#include "lib/settings.h"
#include "meta/pick_type.h"
#include "meta/predicate_for_all_values.h"

namespace integrate {
namespace term_prob {
namespace enum_term_prob {
template <TermProbType type>
struct EnumTermProb : public PickType<type, constant::Constant, n_iters::NIters,
                                      multiplier_func::MultiplierFunc> {};

template <TermProbType type>
struct IsTermProb : BoolWrapper<TermProb<EnumTermProb<type>, Settings<type>>> {
};

static_assert(PredicateForAllValues<TermProbType>::value<IsTermProb>);
} // namespace enum_term_prob
} // namespace term_prob
} // namespace integrate
