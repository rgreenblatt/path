#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "lib/span.h"
#include "rng/rng.h"

namespace rng {
template <typename SG>
concept SequenceGen = requires(SG &state, unsigned dimension, unsigned count) {
  { state(dimension, count) }
  ->std::same_as<Span<const float>>;

  typename SG::Settings;
  requires requires(const typename SG::Settings &settings) {
    { state.init(settings) }
    ->std::same_as<void>;
  };
};

template <ExecutionModel execution_model, template <ExecutionModel> class SGE>
requires SequenceGen<SGE<execution_model>> struct RngFromSequenceGen {
  using SG = SGE<execution_model>;

  struct Ref {
    struct State {
      HOST_DEVICE State() = default;

      HOST_DEVICE State(unsigned sample_idx, const Ref *ref)
          : dim_(0), sample_(sample_idx), ref_(ref) {}

      HOST_DEVICE inline float next() {
        assert(dim_ < ref_->dimension_bound_);
        assert(sample_ < ref_->samples_per_ * ref_->next_vals_);

        float out = ref_->vals_[sample_ * ref_->dimension_bound_ + dim_];

        dim_++;

        if (dim_ >= ref_->dimension_bound_) {
          sample_ += ref_->samples_per_;
          dim_ = 0;

          if (sample_ >= ref_->samples_per_ * ref_->next_vals_) {
            sample_ -= ref_->samples_per_ * ref_->next_vals_;
          }
        }

        return out;
      }

    private:
      unsigned dim_;
      unsigned sample_;
      const Ref *ref_;
    };

    HOST_DEVICE Ref() {}

    Ref(unsigned samples_per, unsigned next_vals, unsigned dimension_bound,
        Span<const float> vals)
        : samples_per_(samples_per), next_vals_(next_vals),
          dimension_bound_(dimension_bound), vals_(vals) {}

    HOST_DEVICE inline State get_generator(unsigned sample_idx, unsigned,
                                           unsigned) const {
      return State(sample_idx, this);
    }

  private:
    unsigned samples_per_;
    unsigned next_vals_;
    unsigned dimension_bound_;
    Span<const float> vals_;
  };

  RngFromSequenceGen() {}

  // Are x_dim and y_dim actually unimportant???
  Ref gen(const typename SG::Settings &settings, unsigned samples_per, unsigned,
          unsigned, unsigned max_sample_size) {
    gen_.init(settings);
    unsigned dimension_bound = max_sample_size;
    unsigned next_vals = 4;
    unsigned count = next_vals * samples_per;

    auto vals = gen_(dimension_bound, count);

    return Ref(samples_per, next_vals, dimension_bound, vals);
  }

private:
  SG gen_;
};
} // namespace rng
