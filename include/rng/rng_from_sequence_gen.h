#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/fnv.h"
#include "lib/settings.h"
#include "lib/span.h"
#include "rng/rng.h"
#include "rng/rng_from_sequence_gen_settings.h"

namespace rng {
namespace detail {
struct SequenceGenOutput {
  Span<const float> vals;
  unsigned initial_dimension_bound;
};

template <typename SG, typename S>
concept SequenceGen = requires(SG &gen, const S &settings, unsigned dimension,
                               unsigned count) {
  requires Setting<S>;
  { gen.gen(settings, dimension, count) } -> std::same_as<SequenceGenOutput>;
};

struct RngFromSequenceGenRef {
  unsigned samples_per;
  unsigned dimension_bound;
  unsigned initial_dimension_bound;
  Span<const float> vals;

  struct SavedState {
    unsigned dim;
  };

  class State {
  public:
    HOST_DEVICE State() = default;

    HOST_DEVICE State(unsigned initial_dim, unsigned sample_idx,
                      const RngFromSequenceGenRef *ref)
        : initial_dim_(initial_dim), dim_(initial_dim_), sample_(sample_idx),
          ref_(ref) {}

    HOST_DEVICE inline float next() {
      debug_assert_assume(ref_ != nullptr);

      float out = ref_->vals[sample_ * ref_->dimension_bound + dim_];

      ++dim_;
      dim_ %= ref_->dimension_bound;

      return out;
    }

    HOST_DEVICE SavedState save() const { return {.dim = dim_}; }

  private:
    unsigned initial_dim_;
    unsigned dim_;
    unsigned sample_;
    const RngFromSequenceGenRef *ref_;

    friend struct RngFromSequenceGenRef;
  };

  ATTR_PURE_NDEBUG HOST_DEVICE inline State
  get_generator(unsigned sample_idx, unsigned location) const {
    debug_assert(sample_idx < samples_per);

    // the hash is used to make different locations look roughly uncorrelated.
    // Note that they may be correlated in practice (depending on the
    // sequence).
    // For path tracing this make pixels look uncorrelated.
    return State(fnv_hash(location) % initial_dimension_bound, sample_idx,
                 this);
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline State
  state_from_saved(unsigned sample_idx, unsigned location,
                   SavedState saved_state) const {
    auto out = get_generator(sample_idx, location);
    out.dim_ = saved_state.dim;
    return out;
  }
};

template <typename SG, Setting S>
requires SequenceGen<SG, S>
class RngFromSequenceGen {
public:
  using Ref = RngFromSequenceGenRef;

  RngFromSequenceGen() {}

  Ref gen(const RngFromSequenceGenSettings<S> &settings, unsigned samples_per,
          unsigned /*n_locations*/) {
    auto [vals, initial_dimension_bound] = gen_.gen(
        settings.sequence_settings, settings.max_sample_size, samples_per);

    return {samples_per, settings.max_sample_size,
            std::min(initial_dimension_bound, settings.max_sample_size), vals};
  }

private:
  SG gen_;
};
} // namespace detail
} // namespace rng
