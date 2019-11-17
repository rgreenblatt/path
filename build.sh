#!/usr/bin/env bash

unbuffer cmake -H. -Brelease -DCMAKE_BUILD_TYPE=Release | bat -p &&
  unbuffer cmake --build release | bat -p
