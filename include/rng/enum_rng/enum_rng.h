#pragma once

#include "execution_model/execution_model.h"
#include "meta/all_values_enum.h"
#include "rng/enum_rng/rng_type.h"
#include "rng/enum_rng/settings.h"
#include "rng/sobel/sobel.h"
#include "rng/uniform/uniform.h"

namespace rng {
namespace enum_rng {
template <RngType type, ExecutionModel execution_model>
struct EnumRng : public PickType<type, uniform::Uniform<execution_model>,
                                 sobel::Sobel<execution_model>> {};

template <RngType type, ExecutionModel exec>
struct IsRng : BoolWrapper<Rng<EnumRng<type, exec>, Settings<type>>> {};

static_assert(PredicateForAllValues<RngType, ExecutionModel>::value<IsRng>);
} // namespace enum_rng
} // namespace rng
