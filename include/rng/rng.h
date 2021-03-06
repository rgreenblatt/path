#pragma once

#include "lib/settings.h"
#include "meta/mock.h"

#include <array>
#include <concepts>

namespace rng {
template <typename T>
concept RngState = requires(T &state) {
  requires std::semiregular<T>;
  { state.next() } -> std::same_as<float>;
};

template <typename T>
concept RngRef = requires(const T &ref, unsigned sample_idx,
                          unsigned location) {
  requires std::copyable<T>;
  typename T::State;
  requires RngState<typename T::State>;
  {
    ref.get_generator(sample_idx, location)
    } -> std::same_as<typename T::State>;
  typename T::SavedState;
  std::copyable<typename T::SavedState>;
  requires requires(const typename T::State &state) {
    { state.save() } -> std::same_as<typename T::SavedState>;
  };
  requires requires(const typename T::SavedState &saved_state) {
    {
      ref.state_from_saved(sample_idx, location, saved_state)
      } -> std::same_as<typename T::State>;
  };
};

template <typename T, typename S>
concept Rng = requires {
  requires Setting<S>;
  typename T::Ref;
  requires RngRef<typename T::Ref>;
  requires requires(T & rng, const S &settings, unsigned samples_per,
                    unsigned n_locations) {
    {
      rng.gen(settings, samples_per, n_locations)
      } -> std::same_as<typename T::Ref>;
  };
};

struct MockRng : MockNoRequirements {
  struct Ref : MockCopyable {
    struct SavedState : MockSemiregular {};
    struct State : MockSemiregular {
      float next();
      SavedState save() const;
    };

    State get_generator(unsigned sample_idx, unsigned location) const;
    State state_from_saved(unsigned sample_idx, unsigned location,
                           const SavedState &state) const;
  };

  Ref gen(const EmptySettings &settings, unsigned samples_per,
          unsigned n_locations);
};

using MockRngRef = MockRng::Ref;
using MockRngState = MockRng::Ref::State;

static_assert(Rng<MockRng, EmptySettings>);
} // namespace rng
