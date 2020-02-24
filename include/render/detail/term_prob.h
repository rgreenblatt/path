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
concept TermProbRef = requires(const V &term_prob,
                               const Eigen::Array3f &multiplier) {
  { term_prob(multiplier) }
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

    HOST_DEVICE float operator()(const Eigen::Array3f &) const { return prob_; }

  private:
    float prob_;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};

template <ExecutionModel execution_model>
struct TermProbImpl<TermProbType::MultiplierNorm, execution_model> {
public:
  using Settings = TermProbSettings<TermProbType::MultiplierNorm>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings) {
      float shifted_scale =
          std::clamp(settings.convexity_scale + 2.0f, 1e-5f, 4 - 1e-5f);
      float start = std::floor(shifted_scale);
      alpha_ = shifted_scale - start;
      idx_ = start;
    }

    HOST_DEVICE float operator()(const Eigen::Array3f &multiplier) const {
      float squared_norm = multiplier.matrix().squaredNorm();

      // potentially save a bit on compute...
      float norm = idx_ >= 2 ? std::sqrt(squared_norm) : 0;

      float quartic_norm = squared_norm * squared_norm;

      auto get_value = [&](const unsigned idx) {
        switch (idx) {
        case 0:
          // very concave:
          // 1 - x^4
          return 1 - quartic_norm;
        case 1:
          // concave:
          // 1 - x^2
          return 1 - squared_norm;
        case 2:
          // linear:
          // 1 - x
          return 1 - norm;
        case 3:
          // convex:
          // (x - 1)^2
          {
            float x_minus_1 = norm - 1;
            x_minus_1 *= x_minus_1;

            return x_minus_1;
          }
        case 4:
        default:

          // very convex:
          // (x - 1)^4
          {
            float x_minus_1 = norm - 1;
            x_minus_1 *= x_minus_1;
            x_minus_1 *= x_minus_1;
            return x_minus_1;
          }
        }
      };

      assert(get_value(0) >= 0.0f);
      assert(get_value(1) >= 0.0f);
      assert(get_value(2) >= 0.0f);
      assert(get_value(3) >= 0.0f);
      assert(get_value(0) <= 1.0f);
      assert(get_value(1) <= 1.0f);
      assert(get_value(2) <= 1.0f);
      assert(get_value(3) <= 1.0f);

      return alpha_ * get_value(idx_ + 1) + (1 - alpha_) * get_value(idx_);
    }

  private:
    unsigned idx_;
    float alpha_;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};
} // namespace detail
} // namespace render
