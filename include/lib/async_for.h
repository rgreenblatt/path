#include <functional>
#include <future>

template <bool is_async, typename F>
void async_for(unsigned start, unsigned end, const F &f) {
  // alternative async strategy may be better...
  if constexpr (is_async) {
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
