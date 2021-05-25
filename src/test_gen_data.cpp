#include "generate_data/gen_data.h"
#include "lib/assert.h"
#include "lib/info/timer.h"

#include <boost/lexical_cast.hpp>
#include <docopt.h>

// In retrospect, I don't really like docopt...
constexpr char USAGE[] =
    R"(test_gen_data

    Usage:
      test_gen_data [--base-seed=<seed>] [--count=<count>]
      test_gen_data (-h | --help)

    Options:
      -h --help              Show this screen.
      --base-seed=<seed>     Starting seed [default: 0]
      --count=<count>        Number of scenes [default: 1024]
)";

int main(int argc, char *argv[]) {
  const std::map<std::string, docopt::value> args =
      docopt::docopt(USAGE, {argv + 1, argv + argc});

  auto get_unpack_arg = [&](const std::string &s) {
    auto it = args.find(s);
    if (it == args.end()) {
      std::cerr << "internal command line parse error" << std::endl;
      std::cerr << s << std::endl;
      unreachable();
    }

    return it->second;
  };

  unsigned base_seed = get_unpack_arg("--base-seed").asLong();
  unsigned count = get_unpack_arg("--count").asLong();

  generate_data::gen_data(1, 1, 1, 0);

  Timer timer;
  generate_data::gen_data(count, 1, 1, base_seed);
  timer.report("run time");

  generate_data::deinit_renderers();
}
