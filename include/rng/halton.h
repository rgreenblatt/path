#pragma once

#include <array>

namespace rng {
namespace halton_detail {
constexpr std::array<unsigned, 11> primes = {1,  2,  3,  5,  7, 11,
                                             13, 17, 19, 23, 29};

template <typename T, std::size_t size>
constexpr T sum(const std::array<T, size> &arr) {
  T ret = T(0);
  for (unsigned i = 0; i < size; ++i) {
    ret += arr[i];
  }

  return ret;
}

template <unsigned dim> constexpr std::array<float, dim> halton(unsigned i) {
  static_assert(primes.size() > dim);

  // source: https://people.sc.fsu.edu/~jburkardt/cpp_src/halton/halton.html
  std::array<float, dim> prime_inv = {};
  std::array<float, dim> r = {};
  std::array<unsigned, dim> t = {};

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

  while (0 < sum(t)) {
    for (unsigned j = 0; j < dim; j++) {
      unsigned d = (t[j] % primes[j + 1]);
      r[j] = r[j] + (float)(d)*prime_inv[j];
      prime_inv[j] = prime_inv[j] / (float)(primes[j + 1]);
      t[j] = (t[j] / primes[j + 1]);
    }
  }

  return r;
}
} // namespace halton_detail

// SPEED: does this matter?
constexpr unsigned sequence_size = 2048;

template <unsigned dim>
constexpr std::array<std::array<float, dim>, sequence_size>
get_halton_sequence() {
  std::array<std::array<float, dim>, sequence_size> arr = {};
  for (unsigned i = 0; i < sequence_size; ++i) {
    arr[i] = halton_detail::halton<dim>(i);
  }

  return arr;
}

template <unsigned dim>
constexpr std::array<std::array<float, dim>, sequence_size>
    halton_sequence = get_halton_sequence<dim>();

template <unsigned dim, bool use_cache = true>
constexpr std::array<float, dim> halton(unsigned i) {
  if (use_cache && i < sequence_size) {
    return halton_sequence<dim>[i];
  } else {
    return halton_detail::halton<dim>(i);
  }
}
} // namespace rng