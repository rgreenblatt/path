# should be sourced

module load gcc/10.2
module load cuda/11.1.1
module load cudnn/8.2.0
module load cmake/3.20.0
module load anaconda/2020.02
module load boost/1.69
conda activate pytorch_env
export BUILD_FOR_PATH_CMAKE_CLI_GEN_ARGS=" -DCMAKE_PREFIX_PATH=$HOME/.conda/envs/pytorch_env/lib/python3.8/site-packages/torch/share/cmake/;$HOME/.conda/envs/pytorch_env/lib/python3.8/site-packages/pybind11/share/cmake/ -DCUDNN_LIBRARY_PATH=/gpfs/runtime/opt/cudnn/8.2.0/lib64/libcudnn.so -DCUDNN_INCLUDE_PATH=/gpfs/runtime/opt/cudnn/8.2.0/include -DCMAKE_CXX_FLAGS='-cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/ -cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/x86_64-pc-linux-gnu/ -D_GLIBCXX_USE_CXX11_ABI=0' -DCMAKE_CUDA_FLAGS='-cxx-isystem /users/rgreenb6/.local/etc/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-16.04/lib/clang/12.0.0/include/cuda_wrappers/ -cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/ -cxx-isystem /gpfs/runtime/opt/gcc/10.2/include/c++/10.2.0/x86_64-pc-linux-gnu/ -cxx-isystem /users/rgreenb6/.local/etc/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-16.04/lib/clang/12.0.0/include/ -D_GLIBCXX_USE_CXX11_ABI=0' -DCMAKE_CXX_STANDARD_LIBRARIES='/gpfs/runtime/opt/gcc/10.2/lib64/libstdc++.so'"
