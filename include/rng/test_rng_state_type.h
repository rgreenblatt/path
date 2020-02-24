#include "rng/rng.h"
#include "rng/uniform.h"

// TODO: hack used test if function can take RngState concept
namespace rng {
using TestRngStateT =
    typename RngT<RngType::Uniform, ExecutionModel::CPU>::Ref::State;

static_assert(RngState<TestRngStateT>);
} // namespace rng
