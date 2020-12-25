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
concept RngRef = requires(const T &ref, unsigned sample_idx,
                          unsigned location) {
  requires std::copyable<T>;
  { ref.get_generator(sample_idx, location) }
  ->RngState;
};

template <typename T, typename S> concept Rng = requires {
  requires Setting<S>;
  requires requires(T & rng, const S &settings, unsigned samples_per,
                    unsigned n_locations) {
    { rng.gen(settings, samples_per, n_locations) }
    ->RngRef;
  };
};

struct MockRngSettings : EmptySettings {};

struct MockRng : MockNoRequirements {
  struct Ref : MockCopyable {
    struct State : MockSemiregular {
      float next();
    };

    State get_generator(unsigned, unsigned) const;
  };

  Ref gen(const MockRngSettings &, unsigned, unsigned);
};

using MockRngRef = MockRng::Ref;
using MockRngState = MockRng::Ref::State;

static_assert(Rng<MockRng, MockRngSettings>);
} // namespace rng
