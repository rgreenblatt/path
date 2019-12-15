# GPU Raytracer
 
## Usage

Building on the department machines is likely to be challenging and I wouldn't
recommended trying to set up the environment needed to build this project.
This project requires clang-9, cuda 10+, and boost. The path to cuda is hard
coded as /usr/local/cuda/ because this is a requirement to use clang as a cuda
compiler. I am not providing a Makefile (the build system is CMake), but the
"build.sh" script can be used to build.

There are several binaries produced in this project. The "final" binary runs
the project. This has a REPL with some basic controls (mostly for debugging).

## Design

I have chosen to focus mostly on performance and I think I have succeeded
in that objective. Due to this, the code is somewhat complex and multiple
data structures are used. This includes a kdtree and a projection based
data structure I won't go into here.
