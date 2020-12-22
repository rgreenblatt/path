#pragma once

#include "rng/rng.h"
#include "rng/uniform.h"

// TODO: hack used to test if function can take RngState concept
namespace rng {
using DummyRngState =
    typename RngT<RngType::Uniform, ExecutionModel::CPU>::Ref::State;

static_assert(RngState<DummyRngState>);
} // namespace rng
