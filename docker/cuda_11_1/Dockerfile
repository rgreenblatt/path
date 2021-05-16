FROM nvcr.io/nvidia/pytorch:20.12-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \ 
	wget \ 
  curl \
	git \
	neovim \
	zip \
	build-essential \
	libstdc++-10-dev \
	qt5-default \
	software-properties-common
RUN conda install numpy tensorboard pip scipy && pip install gitpython

ENV PATH="/root/.local/bin/:${PATH}"
RUN mkdir ~/.local/bin/

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add - && \
	apt-add-repository \
	"deb http://apt.llvm.org/focal/ llvm-toolchain-focal-12 main" && \
	apt-get install -y clang-12 llvm-12 && \
	ln -s /usr/bin/clang++-12 ~/.local/bin/clang++

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
	2>/dev/null | gpg --dearmor - \
	| tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
	apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
	apt-get install -y cmake && ln -s /usr/bin/cmake ~/.local/bin/cmake

WORKDIR /root/
RUN git clone https://github.com/rgreenblatt/cmake_cli && cd cmake_cli && \
	python3 setup.py install && cd .. && rm -rf cmake_cli

RUN git clone https://github.com/rgreenblatt/path && cd path && \
  git submodule update --init --recursive && \
	cp extern/include/cub/util_debug.cuh /usr/local/cuda/include/cub/ && \
	cmake_cli build --target neural_render_generate_data --release \
	--gen-args ' -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch/share/cmake/;/opt/conda/lib/python3.8/site-packages/pybind11/share/cmake/' && \
	python setup.py install
WORKDIR /root/path/neural_render
