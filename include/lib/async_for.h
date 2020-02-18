#pragma once

#include <functional>
#include <future>
#include <vector>

template <typename F>
void async_for(bool is_async, unsigned start, unsigned end, const F &f) {
  // alternative async strategy may be better...
  if (is_async) {
    std::vector<std::future<void>> results(end - start);

    for (unsigned i = start; i < end; i++) {
      results[i] = std::async(std::launch::async, [&] { f(i); });
    }

    for (auto &result : results) {
      result.get();
    }
  } else {
    for (unsigned i = start; i < end; i++) {
      f(i);
    }
  }
}
