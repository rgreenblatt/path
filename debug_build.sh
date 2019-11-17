#!/usr/bin/env bash

unbuffer cmake -Bdebug -DCMAKE_BUILD_TYPE=Debug | bat -p &&
  unbuffer cmake --build debug | bat -p
