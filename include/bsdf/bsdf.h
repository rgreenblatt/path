#pragma once

#include "bsdf/bsdf_sample.h"
#include "meta/decays_to.h"
#include "meta/mock.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace bsdf {
template <typename T>
concept BSDF = requires(const T &bsdf, const UnitVector &incoming_dir,
                        const UnitVector &outgoing_dir,
                        const UnitVector &normal, rng::MockRngState &r) {
  { T::continuous } -> DecaysTo<bool>;
  { T::discrete } -> DecaysTo<bool>;
  // is it a brdf or a full bsdf? (hemisphere vs full sphere)
  { bsdf.is_brdf() } -> DecaysTo<bool>;

  // must have some distribution (note that it can be both)
  requires T::continuous || T::discrete;

  requires requires {
    {
      bsdf.continuous_eval(incoming_dir, outgoing_dir, normal)
      } -> DecaysTo<Eigen::Array3f>;
  } || !T::continuous;

  requires requires {
    { bsdf.prob_continuous(incoming_dir, normal) } -> DecaysTo<float>;
  } || !(T::continuous && T::discrete);

  // perhaps samples should be made part of a separate concept?

  requires requires {
    // samples according to the distribution
    // multiplier is the value of the brdf / probability of the sample,
    // so this is just the brdf coefficient term
    {
      bsdf.continuous_sample(incoming_dir, normal, r)
      } -> DecaysTo<bsdf::BSDFSample>;
  } || !T::continuous;

  requires requires {
    { bsdf.discrete_sample(incoming_dir, normal, r) } -> DecaysTo<BSDFSample>;
  } || !T::discrete;
};

template <typename T>
concept ContinuousBSDF = BSDF<T> && T::continuous;

// other possible functions to put in a different concept which might allow for
// clever sampling strategies:
//  - discrete_arr -> returns arr of possible discrete locations and
//    corresponding weights
//  - some sort of summary of the bsdf?

struct MockContinuousBSDF : MockNoRequirements {
  static constexpr bool continuous = true;
  static constexpr bool discrete = false;

  bool is_brdf() const;

  Eigen::Array3f continuous_eval(const UnitVector &, const UnitVector &,
                                 const UnitVector &) const;

  template <rng::RngState R>
  BSDFSample continuous_sample(const UnitVector &, const UnitVector &,
                               R &) const;
};

struct MockDiscreteBSDF : MockNoRequirements {
  static constexpr bool continuous = false;
  static constexpr bool discrete = true;

  bool is_brdf() const;

  template <rng::RngState R>
  BSDFSample discrete_sample(const UnitVector &, const UnitVector &, R &) const;
};

struct MockContinuousDiscreteBSDF : public MockContinuousBSDF,
                                    public MockDiscreteBSDF {
  static constexpr bool continuous = true;
  static constexpr bool discrete = true;

  bool is_brdf() const;

  float prob_continuous(const UnitVector &, const UnitVector &) const;
};

static_assert(ContinuousBSDF<MockContinuousBSDF>);
static_assert(ContinuousBSDF<MockContinuousDiscreteBSDF>);
static_assert(BSDF<MockContinuousBSDF>);
static_assert(BSDF<MockDiscreteBSDF>);
static_assert(BSDF<MockContinuousDiscreteBSDF>);
} // namespace bsdf
