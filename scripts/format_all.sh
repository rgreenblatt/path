#!/usr/bin/env bash

fd -e cpp -e cu -e h -e cuh | xargs clang-format-9 -i
