#include "generate_data/full_scene/generate_data.h"
#include "lib/assert.h"
#include "lib/info/timer.h"

#include <boost/lexical_cast.hpp>
#include <docopt.h>

// In retrospect, I don't really like docopt...
constexpr char USAGE[] =
    R"(test_gen_data

    Usage:
      test_gen_data [--base-seed=<seed>] [--max-tris=<count>]
      test_gen_data (-h | --help)

    Options:
      -h --help              Show this screen.
      --base-seed=<seed>     Starting seed [default: 0]
      --max-tris=<count>     Max number of totall triangles [default: 64]
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
  unsigned max_tris = get_unpack_arg("--max-tris").asLong();

  // generate_data::full_scene::generate_data(200, 0, 0, 0);

  Timer timer;
  generate_data::full_scene::generate_data(max_tris, 1, 1, 5, base_seed);

  generate_data::full_scene::generate_data_for_image(max_tris, 64, 4, 5,
                                                     base_seed);
  timer.report("run time");

  generate_data::full_scene::deinit_renderers();
}
