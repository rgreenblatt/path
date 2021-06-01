#pragma once

#include "boost/hana/range.hpp"
#include "boost/hana/unpack.hpp"
#include "bsdf/bsdf.h"
#include "bsdf/bsdf_sample.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "lib/search_inclusive.h"
#include "lib/span.h"
#include "meta/all_values/sequential_dispatch.h"
#include "rng/rng.h"

#include <numeric>

namespace bsdf {
template <BSDF... T> class Combined {
public:
  constexpr Combined() = default;
  constexpr Combined(MetaTuple<T...> items,
                     const std::array<float, sizeof...(T)> &weights)
      : items_(std::move(items)) {
    debug_assert(std::abs(std::accumulate(weights.begin(), weights.end(), 0.f) -
                          1.f) < 1e-6);

    auto compute_total = [&](SpanSized<const unsigned> idxs,
                             SpanSized<float> inclusive_values) {
      float total = 0.f;
      for (unsigned i = 0; i < idxs.size(); ++i) {
        unsigned idx = idxs[i];
        total += weights[idx];
        // skip last
        if (i < inclusive_values.size()) {
          inclusive_values[i] = total;
        }
      }
      for (float &v : inclusive_values) {
        // we want prob given this case (discrete or continuous)
        v /= total;
      }
    };

    if constexpr (discrete) {
      compute_total(discrete_idxs, weight_discrete_inclusive_);
    }
    if constexpr (continuous) {
      compute_total(continuous_idxs, weight_continuous_inclusive_);
    }

    if constexpr (discrete && continuous) {
      fixed_prob_continuous_ = 0.f;
      for (unsigned idx : only_continuous_idxs) {
        fixed_prob_continuous_ += weights[idx];
      }

      for (unsigned i = 0; i < discrete_and_continuous_idxs.size(); ++i) {
        unsigned idx = discrete_and_continuous_idxs[i];
        weight_discrete_and_continuous_[i] = weights[idx];
      }
    }
  }

  static constexpr bool discrete = (... || T::discrete);
  static constexpr bool continuous = (... || T::continuous);

  ATTR_PURE constexpr bool is_brdf() const {
    return boost::hana::unpack(
        items_, [&](const auto &...v) { return (... && v.is_brdf()); });
  }

  ATTR_PURE_NDEBUG HOST_DEVICE FloatRGB continuous_eval(
      const UnitVector &incoming_dir, const UnitVector &outgoing_dir,
      const UnitVector &normal) const requires(continuous) {
    return boost::hana::unpack(
        boost::hana::range_c<unsigned, 0, continuous_idxs.size()>,
        [&](const auto &...idxs) -> FloatRGB {
          auto get = [&](auto tag_idx) -> FloatRGB {
            constexpr unsigned idx = tag_idx;
            constexpr unsigned overall_idx = continuous_idxs[idx];
            auto out = weight(idx, weight_continuous_inclusive_) *
                       items_[boost::hana::int_c<overall_idx>].continuous_eval(
                           incoming_dir, outgoing_dir, normal);
            return out;
          };
          return (... + get(idxs));
        });
  }

  HOST_DEVICE float prob_continuous(const UnitVector &incoming_dir,
                                    const UnitVector &normal) const
      requires(discrete &&continuous) {
    return boost::hana::unpack(
        boost::hana::range_c<unsigned, 0, discrete_and_continuous_idxs.size()>,
        [&](const auto &...idxs) -> float {
          auto get = [&](auto tag_idx) -> float {
            constexpr unsigned idx = tag_idx;
            constexpr unsigned overall_idx = discrete_and_continuous_idxs[idx];
            return weight_discrete_and_continuous_[idx] *
                   items_[boost::hana::int_c<overall_idx>].prob_continuous(
                       incoming_dir, normal);
          };
          return (fixed_prob_continuous_ + ... + get(idxs));
        });
  }

  template <rng::RngState R>
  HOST_DEVICE BSDFSample continuous_sample(const UnitVector &incoming_dir,
                                           const UnitVector &normal,
                                           R &rng) const requires(continuous) {
    float loc = rng.next();

    unsigned idx = search_inclusive(loc, weight_continuous_inclusive_,
                                    binary_search_threshold, true);

    return sequential_dispatch<continuous_idxs.size()>(idx, [&](auto tag_idx) {
      constexpr unsigned idx = tag_idx;
      constexpr unsigned overall_idx = continuous_idxs[idx];
      return items_[boost::hana::int_c<overall_idx>].continuous_sample(
          incoming_dir, normal, rng);
    });
  }

  template <rng::RngState R>
  HOST_DEVICE BSDFSample discrete_sample(const UnitVector &incoming_dir,
                                         const UnitVector &normal, R &rng) const
      requires(discrete) {
    float loc = rng.next();

    unsigned idx = search_inclusive(loc, weight_discrete_inclusive_,
                                    binary_search_threshold, true);

    return sequential_dispatch<discrete_idxs.size()>(idx, [&](auto tag_idx) {
      constexpr unsigned idx = tag_idx;
      constexpr unsigned overall_idx = discrete_idxs[idx];
      return items_[boost::hana::int_c<overall_idx>].discrete_sample(
          incoming_dir, normal, rng);
    });
  }

private : template <template <typename> class Getter> struct Idxs {
    static constexpr unsigned count = (... + (Getter<T>::value ? 1 : 0));
    static_assert(count <= sizeof...(T));
    static constexpr std::array<unsigned, count> idxs = []() {
      std::array<unsigned, count> out;
      unsigned idx = 0;
      unsigned overall = 0;

      auto add_idx = [&](bool value) {
        if (value) {
          out[idx] = overall;
          ++idx;
        }
        ++overall;
      };

      (add_idx(Getter<T>::value), ...);

      always_assert(idx == count);
      always_assert(overall == sizeof...(T));

      return out;
    }();
  };

  template <typename SubT>
  using IsDiscrete = std::bool_constant<SubT::discrete>;
  template <typename SubT>
  using IsContinuous = std::bool_constant<SubT::continuous>;
  template <typename SubT>
  using IsDiscreteAndContinuous =
      std::bool_constant<SubT::discrete && SubT::continuous>;
  template <typename SubT>
  using IsOnlyContinuous =
      std::bool_constant<SubT::continuous && (!SubT::discrete)>;

  static constexpr auto discrete_idxs = Idxs<IsDiscrete>::idxs;
  static constexpr auto continuous_idxs = Idxs<IsContinuous>::idxs;
  static constexpr auto discrete_and_continuous_idxs =
      Idxs<IsDiscreteAndContinuous>::idxs;
  static constexpr auto only_continuous_idxs = Idxs<IsOnlyContinuous>::idxs;

  static_assert(discrete_idxs.size() + continuous_idxs.size() >= sizeof...(T));

  constexpr float mass_inclusive(unsigned i, SpanSized<const float> arr) const {
    if (i >= arr.size()) {
      return 1.f;
    } else {
      return arr[i];
    }
  }

  constexpr float mass_exclusive(unsigned i, Span<const float> arr) const {
    if (i == 0) {
      return 0.f;
    } else {
      return arr[i - 1];
    }
  }

  constexpr float weight(unsigned i, SpanSized<const float> arr) const {
    return mass_inclusive(i, arr) - mass_exclusive(i, arr);
  }

  // TODO!
  static constexpr unsigned binary_search_threshold = 4096;

  [[no_unique_address]] MetaTuple<T...> items_;

  struct EmptyT {};

  // We could compute the inclusive sums on demand and make this size
  // somewhat smaller...

  [[no_unique_address]] std::conditional_t<
      continuous, std::array<float, continuous_idxs.size() - 1>, EmptyT>
      weight_continuous_inclusive_;
  [[no_unique_address]] std::conditional_t<
      discrete, std::array<float, discrete_idxs.size() - 1>, EmptyT>
      weight_discrete_inclusive_;
  // NOTE: not an inclusive sum!
  [[no_unique_address]] std::conditional_t<
      discrete && continuous,
      std::array<float, discrete_and_continuous_idxs.size()>, EmptyT>
      weight_discrete_and_continuous_;

  // total probability of bsdfs which are only continuous
  [[no_unique_address]] std::conditional_t<discrete && continuous, float,
                                           EmptyT>
      fixed_prob_continuous_;
};
} // namespace bsdf
