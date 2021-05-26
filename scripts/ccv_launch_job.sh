#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 4 CPU cores
#SBATCH -n 4

#SBATCH -t 18:00:00
#SBATCH -o train.out

source scripts/ccv_setup.sh
./scripts/build_ccv.sh

cd neural_render && pwd && python3 src/train.py --name ray_input
