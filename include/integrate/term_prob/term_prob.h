#pragma once

#include "lib/settings.h"
#include "meta/mock.h"

#include <Eigen/Core>

namespace integrate {
namespace term_prob {
template <typename T>
concept TermProbRef = requires(const T &term_prob, unsigned iters,
                               const Eigen::Array3f &multiplier) {
  requires std::copyable<T>;
  { term_prob(iters, multiplier) } -> std::convertible_to<float>;
};

template <typename T, typename S>
concept TermProb = requires(T &term_prob, const S &settings) {
  requires Setting<S>;
  requires std::movable<T>;
  requires std::default_initializable<T>;

  { term_prob.gen(settings) } -> TermProbRef;
};

struct MockTermProb : MockDefaultInitMovable {
  struct Ref : MockCopyable {
    float operator()(unsigned, const Eigen::Array3f &) const;
  };

  Ref gen(const EmptySettings &);
};

static_assert(TermProb<MockTermProb, EmptySettings>);
} // namespace term_prob
} // namespace integrate
