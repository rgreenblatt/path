#include <functional>

void parallel_for_loop(unsigned start, unsigned end,
                       std::function<void(unsigned)> f);
