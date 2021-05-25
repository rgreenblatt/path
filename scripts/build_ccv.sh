#!/usr/bin/env bash

# calling this hacky would be a huge understatement...

cmake_cli build --target neural_render_generate_data --release --gen-args "$BUILD_FOR_PATH_CMAKE_CLI_GEN_ARGS" --cuda-archs '60;75;80' && python3 setup.py install --user
