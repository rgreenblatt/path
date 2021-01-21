#pragma once

#include "execution_model/execution_model.h"
#include "lib/span.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/predicate_for_all_values.h"
#include "rng/rng.h"
#include "rng/rng_from_sequence_gen.h"
#include "rng/sobel/settings.h"

#include <memory>

namespace rng {
namespace sobel {
namespace detail {
template <ExecutionModel exec> class SobelSequenceGen {
public:
  // need to implementated when Generator is defined
  SobelSequenceGen();
  ~SobelSequenceGen();
  SobelSequenceGen(SobelSequenceGen &&);
  SobelSequenceGen &operator=(SobelSequenceGen &&);

  rng::detail::SequenceGenOutput gen(const SobelSettings &settings,
                                     unsigned dimension_bound, unsigned count);

private:
  // PIMPL
  class Generator;

  std::unique_ptr<Generator> gen_;
};
} // namespace detail

template <ExecutionModel exec>
using Sobel = rng::detail::RngFromSequenceGen<detail::SobelSequenceGen<exec>,
                                              detail::SobelSettings>;

template <ExecutionModel exec>
struct IsRng : std::bool_constant<Rng<Sobel<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsRng>);
} // namespace sobel
} // namespace rng
