#pragma once

#include "execution_model/thrust_data.h"
#include "lib/span.h"
#include "rng/rng_from_sequence_gen.h"

namespace rng {
namespace halton_detail {
constexpr std::array<unsigned, 11> primes = {1,  2,  3,  5,  7, 11,
                                             13, 17, 19, 23, 29};

template <typename T> constexpr T sum(SpanSized<const T> s) {
  T ret = T(0);
  for (unsigned i = 0; i < s.size(); ++i) {
    ret += s[i];
  }

  return ret;
}

constexpr void halton(unsigned i, Span<float> f_working_mem,
                      Span<unsigned> u_working_mem, SpanSized<float> values) {
  unsigned dim = values.size();

  // source: https://people.sc.fsu.edu/~jburkardt/cpp_src/halton/halton.html
  auto prime_inv = f_working_mem;
  auto r = values;
  auto t = u_working_mem.slice(0, dim);

  for (unsigned j = 0; j < dim; j++) {
    t[j] = i;
  }

  //
  //  Carry out the computation.
  //
  for (unsigned j = 0; j < dim; j++) {
    prime_inv[j] = 1.0 / (float)(primes[j + 1]);
  }

  for (unsigned j = 0; j < dim; j++) {
    r[j] = 0.0;
  }

  while (0 < sum<unsigned>(t)) {
    for (unsigned j = 0; j < dim; j++) {
      unsigned d = (t[j] % primes[j + 1]);
      r[j] = r[j] + (float)(d)*prime_inv[j];
      prime_inv[j] = prime_inv[j] / (float)(primes[j + 1]);
      t[j] = (t[j] / primes[j + 1]);
    }
  }
}

template <ExecutionModel execution_model> struct HaltonSequenceGen {
  Span<const float> operator()(unsigned dimension_bound, unsigned count) {
    unsigned size = dimension_bound * count;
    f_working_mem_.resize(size);
    u_working_mem_.resize(size);
    vals_.resize(size);

    Span<float> f_working_mem = f_working_mem_;
    Span<unsigned> u_working_mem = u_working_mem_;
    Span<float> vals = vals_;

    // SPEED: memory could be reused more efficiently/tiling...
    auto start_it = thrust::make_counting_iterator(0u);
    thrust::for_each(thrust_data_.execution_policy(), start_it,
                     start_it + count, [=] HOST_DEVICE(unsigned i) {
                       const unsigned start = i * dimension_bound;
                       const unsigned end = (i + 1) * dimension_bound;
                       halton(i, f_working_mem.slice(start, end),
                              u_working_mem.slice(start, end),
                              vals.slice(start, end));
                     });

    return vals_;
  }

  using Settings = rng::RngSettings<RngType::Halton>;

  void init(const Settings &) {}

private:
  ThrustData<execution_model> thrust_data_;

  ExecVector<execution_model, float> f_working_mem_;
  ExecVector<execution_model, unsigned> u_working_mem_;
  ExecVector<execution_model, float> vals_;
};
} // namespace halton_detail

template <ExecutionModel execution_model>
struct RngImpl<RngType::Halton, execution_model>
    : RngFromSequenceGen<execution_model, halton_detail::HaltonSequenceGen> {
  using Ref =
      typename RngFromSequenceGen<execution_model,
                                  halton_detail::HaltonSequenceGen>::Ref;
};

static_assert(Rng<RngType::Halton, ExecutionModel::GPU>);
static_assert(Rng<RngType::Halton, ExecutionModel::CPU>);
} // namespace rng
