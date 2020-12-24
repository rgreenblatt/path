#pragma once

#include "lib/cuda/curand_utils.h"
#include "lib/cuda/utils.h"
#include "lib/settings.h"
#include "lib/span.h"
#include "lib/utils.h"
#include "rng/rng.h"

#include <curand_kernel.h>

namespace rng {
namespace detail {
constexpr unsigned fnv_hash(unsigned in) {
  constexpr unsigned prime = 16777619;
  constexpr unsigned offset_basis = 2166136261;
  unsigned out_hash = offset_basis;
  for (unsigned i = 0; i < 4; ++i) {
    out_hash ^= (in >> (8 * i)) & 0xff;
    out_hash *= prime;
  }
  return out_hash;
}
} // namespace detail

template <typename SG, typename S>
concept SequenceGen = requires(SG &state, unsigned dimension, unsigned count,
                               const S &settings) {
  requires Setting<S>;
  { state(dimension, count) }
  ->std::same_as<Span<const float>>;

  state.init(settings);
};

template <typename SG, Setting S>
requires SequenceGen<SG, S> struct RngFromSequenceGen {
  struct Ref {
    unsigned samples_per_;
    unsigned dimension_bound_;
    Span<const float> vals_;

    class State {
    public:
      HOST_DEVICE State() = default;

      HOST_DEVICE State(unsigned initial_dim, unsigned sample_idx,
                        const Ref *ref)
          : initial_dim_(initial_dim), dim_(initial_dim_), sample_(sample_idx),
            ref_(ref) {}

      HOST_DEVICE inline float next() {
        assert(ref_ != nullptr);

        float out = ref_->vals_[sample_ * ref_->dimension_bound_ + dim_];

        ++dim_;
        dim_ %= ref_->dimension_bound_;

        return out;
      }

    private:
      unsigned initial_dim_;
      unsigned dim_;
      unsigned sample_;
      const Ref *ref_;
    };

    HOST_DEVICE inline State get_generator(unsigned sample_idx,
                                           unsigned location) const {
      assert(sample_idx < samples_per_);

      // the hash is used to make different locations look roughly uncorrelated.
      // Note that they may be correlated in practice (depending on the
      // sequence).
      // For path tracing this make pixels look uncorrelated.
      return State(detail::fnv_hash(location) % dimension_bound_, sample_idx,
                   this);
    }
  };

  RngFromSequenceGen() {}

  Ref gen(const S &settings, unsigned samples_per, unsigned /*n_locations*/,
          unsigned max_sample_size) {
    gen_.init(settings);
    unsigned dimension_bound = max_sample_size;

    auto vals = gen_(dimension_bound, samples_per);

    return Ref{samples_per, dimension_bound, vals};
  }

private:
  SG gen_;
};
} // namespace rng
