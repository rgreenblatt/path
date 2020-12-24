#pragma once

#include "lib/settings.h"
#include "meta/mock.h"

#include <array>
#include <concepts>

namespace rng {
template <typename T> concept RngState = requires(T &state) {
  requires std::semiregular<T>;
  { state.next() }
  ->std::same_as<float>;
};

template <typename T>
concept RngRef = requires(const T &ref, unsigned sample_idx, unsigned x,
                          unsigned y) {
  requires std::copyable<T>;
  { ref.get_generator(x, y, sample_idx) }
  ->RngState;
};

template <typename T, typename S> concept Rng = requires {
  requires Setting<S>;
  requires requires(T & rng, const S &settings, unsigned samples_per,
                    unsigned x_dim, unsigned y_dim,
                    unsigned max_draws_per_sample) {
    { rng.gen(settings, samples_per, x_dim, y_dim, max_draws_per_sample) }
    ->RngRef;
  };
};

struct MockRngSettings : EmptySettings {};

struct MockRng : MockNoRequirements {
  struct Ref : MockCopyable {
    struct State : MockSemiregular {
      float next();
    };

    State get_generator(unsigned, unsigned, unsigned) const;
  };

  Ref gen(const MockRngSettings &, unsigned, unsigned, unsigned, unsigned);
};

using MockRngRef = MockRng::Ref;
using MockRngState = MockRng::Ref::State;

static_assert(Rng<MockRng, MockRngSettings>);
} // namespace rng
