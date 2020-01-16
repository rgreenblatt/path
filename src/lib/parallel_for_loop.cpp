#include "lib/parallel_for_loop.h"

#include <dbg.h>
#include <omp.h>

void parallel_for_loop(unsigned start, unsigned end,
                       std::function<void(unsigned)> f) {
#pragma omp parallel for
  for (unsigned i = start; i < end; i++) {
    f(i);
  }
}
