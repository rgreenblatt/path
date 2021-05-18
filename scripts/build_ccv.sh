#!/usr/bin/env bash

# calling this hacky would be a huge understatement...

source scripts/ccv_setup.sh

cmake_cli build --target neural_render_generate_data --release --gen-args " -DCMAKE_PREFIX_PATH=$HOME/.conda/envs/pytorch_env/lib/python3.8/site-packages/torch/share/cmake/;$HOME/.conda/envs/pytorch_env/lib/python3.8/site-packages/pybind11/share/cmake/ -DCUDNN_LIBRARY_PATH=/gpfs/runtime/opt/cudnn/8.2.0/lib64/libcudnn.so -DCUDNN_INCLUDE_PATH=/gpfs/runtime/opt/cudnn/8.2.0/include -DCMAKE_CXX_FLAGS='-cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/ -cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/x86_64-pc-linux-gnu/' -DCMAKE_CUDA_FLAGS='-cxx-isystem /users/rgreenb6/.local/etc/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-16.04/lib/clang/12.0.0/include/cuda_wrappers/ -cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/ -cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/x86_64-pc-linux-gnu/ -cxx-isystem /users/rgreenb6/.local/etc/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-16.04/lib/clang/12.0.0/include/'" && python3 setup.py install --user