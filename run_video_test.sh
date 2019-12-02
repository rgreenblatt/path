#!/usr/bin/env bash

rm -rf gpu_video && mkdir gpu_video && ./build.sh && 
  ./release/video_test "$@"
