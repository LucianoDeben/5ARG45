#!/bin/bash
salloc --time=3:00:00 --partition=gpu_mig -N 1 --gres=gpu:1 --reservation=jhs_tue2022

# Check if salloc was successful
if [ $? -eq 0 ]; then
  # Load necessary modules
  module load 2023
  module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

  source ./venv/bin/activate

  # Check if already in src directory otherwise move there
  if [ ! -d "src" ]; then
    cd src
  fi
else
  echo "salloc failed. No GPU node assigned."
fi