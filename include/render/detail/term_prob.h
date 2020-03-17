#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "material/material.h"
#include "render/term_prob.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <TermProbType type, ExecutionModel execution_model>
struct TermProbImpl;

template <typename V>
concept TermProbRef = requires(const V &term_prob, const unsigned &iters,
                               const Eigen::Array3f &multiplier) {
  { term_prob(iters, multiplier) }
  ->std::convertible_to<float>;
};

template <TermProbType type, ExecutionModel execution_model>
concept TermProb = requires {
  typename TermProbSettings<type>;
  typename TermProbImpl<type, execution_model>;

  requires requires(TermProbImpl<type, execution_model> & term_prob,
                    const TermProbSettings<type> &settings) {
    { term_prob.gen(settings) }
    ->TermProbRef;
  };
};

template <TermProbType type, ExecutionModel execution_model>
requires TermProb<type, execution_model> struct TermProbT
    : TermProbImpl<type, execution_model> {
  using TermProbImpl<type, execution_model>::TermProbImpl;
};

template <ExecutionModel execution_model>
struct TermProbImpl<TermProbType::Constant, execution_model> {
public:
  using Settings = TermProbSettings<TermProbType::Constant>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings) : prob_(settings.prob) {}

    HOST_DEVICE float operator()(unsigned, const Eigen::Array3f &) const {
      return prob_;
    }

  private:
    float prob_;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};

template <ExecutionModel execution_model>
struct TermProbImpl<TermProbType::MultiplierFunc, execution_model> {
public:
  using Settings = TermProbSettings<TermProbType::MultiplierFunc>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings)
        : exp(settings.exp), min_prob(settings.min_prob) {}

    HOST_DEVICE float operator()(unsigned,
                                 const Eigen::Array3f &multiplier) const {
      // normalization (clamp to deal with cases where multiplier may be
      // negative)
      float squared_norm = std::clamp(
          ((multiplier / (multiplier + 1)) * 0.57).matrix().squaredNorm(), 0.0f,
          1.0f);

      float term_prob = std::abs(std::pow(1 - squared_norm, exp));

      return std::max(term_prob, min_prob);
    }

  private:
    float exp;
    float min_prob;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};

template <ExecutionModel execution_model>
struct TermProbImpl<TermProbType::NIters, execution_model> {
public:
  using Settings = TermProbSettings<TermProbType::NIters>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings) : iters_(settings.iters) {}

    HOST_DEVICE float operator()(unsigned iters, const Eigen::Array3f &) const {
      if (iters >= iters_) {
        return 1.0f;
      } else {
        return 0.0f;
      }
    }

  private:
    unsigned iters_;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};
} // namespace detail
} // namespace render
