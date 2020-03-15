#pragma once

#include "execution_model/execution_model.h"
#include "lib/cuda/utils.h"
#include "lib/settings.h"

#include <array>
#include <concepts>

namespace rng {
enum class RngType { Uniform, Halton /*, Sobel*/ };

template <RngType type, ExecutionModel execution_model> struct RngImpl;

template <RngType type> struct RngSettings;

template <> struct RngSettings<RngType::Uniform> : EmptySettings {};
/* template <> struct RngSettings<RngType::Sobel> : EmptySettings {}; */
template <> struct RngSettings<RngType::Halton> : EmptySettings {};

static_assert(Setting<RngSettings<RngType::Uniform>>);
static_assert(Setting<RngSettings<RngType::Halton>>);

template <typename State> concept RngState = requires(State &state) {
  std::default_initializable<State>;
  { state.next() }
  ->std::same_as<float>;
};

template <typename Ref>
concept RngRef = requires(const Ref &ref, unsigned sample_idx, unsigned x,
                          unsigned y) {
  typename Ref::State;
  RngState<typename Ref::State>;
  { ref.get_generator(x, y, sample_idx) }
  ->std::common_with<typename Ref::State>;
};

template <RngType type, ExecutionModel execution_model> concept Rng = requires {
  typename RngImpl<type, execution_model>;
  typename RngImpl<type, execution_model>::Ref;
  RngRef<typename RngImpl<type, execution_model>::Ref>;
  typename RngImpl<type, execution_model>::Ref::State;
  RngState<typename RngImpl<type, execution_model>::Ref::State>;
  typename RngSettings<type>;
  typename RngSettings<type>;

  // generation
  requires requires(RngImpl<type, execution_model> & rng,
                    const RngSettings<type> &settings, unsigned samples_per,
                    unsigned x_dim, unsigned y_dim,
                    unsigned max_draws_per_sample) {
    { rng.gen(settings, samples_per, x_dim, y_dim, max_draws_per_sample) }
    ->std::common_with<typename RngImpl<type, execution_model>::Ref>;
  };
};

template <RngType type, ExecutionModel execution_model>
requires Rng<type, execution_model> struct RngT
    : RngImpl<type, execution_model> {
  HOST_DEVICE RngT() = default;

  using RngImpl<type, execution_model>::RngImpl;
};
} // namespace rng
